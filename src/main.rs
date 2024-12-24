#[macro_use]
extern crate derive_deref;

use fxhash::{FxHashMap, FxHashSet};
use model::{Network, PreGraph, ProcessedInput, Route};
use petgraph::{algo::{is_cyclic_directed, tarjan_scc}, graph::EdgeIndex, visit::EdgeRef, Graph};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use std::{collections::BTreeMap, error::Error, sync::atomic::Ordering};
use ustr::UstrMap;

mod input_inet;
mod input_json;
mod model;
mod output_inet;

const END_BREAKLOOP: bool = false;

enum InputType<'a> {
    Inet(&'a str, &'a str),
    Json(&'a str),
}
enum OutputType<'a> {
    Inet(&'a str),
    Console,
}

fn parse_args<'a>(
    mut args: impl Iterator<Item = &'a str>,
) -> Result<(InputType<'a>, OutputType<'a>), Box<dyn Error>> {
    let mut input_type = None;
    let mut output_type = None;
    while let Some(arg) = args.next() {
        match arg {
            "--input-inet" => {
                let file = args.next().ok_or("Missing input file for --input-inet")?;
                input_type = Some(InputType::Inet(file, ""));
            }
            "--sequence" => {
                let seq = args.next().ok_or("Missing sequence for --sequence")?;
                input_type = match input_type {
                    Some(InputType::Inet(file, _)) => Some(InputType::Inet(file, seq)),
                    _ => return Err("--sequence can only be used with --input-inet".into()),
                };
            }
            "--input-json" => {
                let file = args.next().ok_or("Missing input file for --input-json")?;
                input_type = Some(InputType::Json(file));
            }
            "--output-inet" => {
                let file = args.next().ok_or("Missing output file for --output-inet")?;
                output_type = Some(OutputType::Inet(file));
            }
            "--output-console" => {
                output_type = Some(OutputType::Console);
            }
            arg => return Err(format!("Unknown argument: {}", arg).into()),
        }
    }

    match (input_type, output_type) {
        (Some(input), Some(output)) => Ok((input, output)),
        _ => Err("Missing input or output type".into()),
    }
}

fn main() {
    let args: SmallVec<[String; 8]> = std::env::args().collect();

    let (input_type, output_type) = parse_args(args.iter().map(AsRef::as_ref).skip(1)).unwrap();

    let processed_input @ ProcessedInput {
        devices,
        links,
        flows,
        ..
    } = match input_type {
        InputType::Inet(filename, sequence) => input_inet::process(filename, sequence),
        InputType::Json(filename) => input_json::process(filename),
    };

    let mut flow_sequence: BTreeMap<u32, UstrMap<_>> = BTreeMap::new();

    for (id, flow) in flows.iter() {
        flow_sequence
            .entry(flow.sequence)
            .or_default()
            .insert(*id, flow);
    }

    // 构建拓扑
    let mut network = Network::default();

    let device_id = devices
        .iter()
        .map(|(name, device)| (*name, network.add_node(device)))
        .collect::<FxHashMap<_, _>>();

    let _link_id = links
        .iter()
        .map(|((from, to), ln)| {
            (
                (*from, *to),
                network.add_edge(device_id[from], device_id[to], ln),
            )
        })
        .collect::<FxHashMap<_, _>>();

    let mut start_offset = 0;

    for (sequence, seq_flows) in flow_sequence {
        let no_seq = sequence == u32::MAX;

        // 构建先序关系图
        let mut pre_graph = PreGraph::default();

        let link_id = links
            .iter()
            .map(|(id, ln)| (*id, pre_graph.add_node(ln)))
            .collect::<FxHashMap<_, _>>();

        let mut routes = FxHashMap::default();

        for flow in seq_flows.values() {
            // 为流设置最早起始时间
            flow.start_offset.store(start_offset, Ordering::Relaxed);

            for route @ (hop, next_hop) in &flow.routes {
                pre_graph.add_edge(link_id[hop], link_id[next_hop], Route { id: *route });

                // 映射路由到网络流量序列
                routes.entry(route).or_insert_with(Vec::new).push(flow);
            }
        }

        let mut breakloop = Vec::new();

        while pre_graph.node_count() != 0 {
            // 找到所有入度为0的节点
            let queue: SmallVec<[_; 64]> = pre_graph
                .node_indices()
                .filter(|node| {
                    pre_graph
                        .neighbors_directed(*node, petgraph::Incoming)
                        .count()
                        == 0
                })
                .map(|node| pre_graph[node])
                .collect();

            if queue.is_empty() {
                println!("breakloop");
                if END_BREAKLOOP {
                    let mut urgency = Vec::with_capacity(seq_flows.len());
                    urgency.extend(seq_flows
                        .values()
                        .filter(|flow| !flow.schedule_done())
                        .map(|&flow| (flow, flow.calculate_urgency(flow.first_unscheduled_link()).unwrap()))
                    );
                    urgency.sort_unstable_by_key(|(_, urgency)| *urgency);

                    while is_cyclic_directed(&pre_graph.graph) {
                        // 选择紧急度最低的流进行破环
                        let blp @ (flow, _) = urgency.pop().unwrap();
    
                        // 记录被破环的流
                        breakloop.push(blp);
    
                        // 执行破环
                        pre_graph.breakloop(flow);
                    }
                } else {
                    // 循环直到图中没有环
                    while let Some(cycle_edges) = find_cycle_edges(&pre_graph.graph) {
                        // 找到所有造成环的边，统计每个流包含的环边数量
                        let mut flow_to_cycle_edge_count = UstrMap::default();
                        flow_to_cycle_edge_count.reserve(seq_flows.len());

                        for edge in &cycle_edges {
                            for flow in routes[&pre_graph.edge_weight(*edge).unwrap().id]
                                .iter()
                                .filter(|f| !f.breakloop())
                            {
                                *flow_to_cycle_edge_count.entry(flow.id).or_insert(0u32) += 1;
                            }
                        }

                        // 选择被破环的流
                        let flow = flow_to_cycle_edge_count
                            .iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(id, _)| &flows[id])
                            .unwrap();

                        // 记录破环的流
                        breakloop.push((
                            flow,
                            flow.calculate_urgency(flow.first_unscheduled_link())
                                .unwrap(),
                        ));

                        // 执行破环操作
                        pre_graph.breakloop(flow);
                    }

                    breakloop.sort_unstable_by_key(|(_, urgency)| *urgency);

                    for (f, _) in breakloop.iter() {
                        for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                            start_offset = start_offset.max(f.schedule_link(flows.values(), link));
                        }
                    }

                    // 清空破环列表，准备下一轮处理
                    breakloop.clear();
                }
            } else {
                const PARALLEL_THRESHOLD: usize = 8;

                let schedule_link = |link: &&model::Link| {
                    // 获取当前链路待调度的流
                    let mut flows_to_sched: SmallVec<[_; 64]> = link
                        .flows
                        .iter()
                        .map(|f| &flows[f])
                        .filter(|f| {
                            seq_flows.contains_key(&f.id) && !f.schedule_done() && !f.breakloop()
                        })
                        .map(|f| (f, f.calculate_urgency(link).unwrap()))
                        .collect();

                    // 按紧急程度排序
                    flows_to_sched.sort_unstable_by_key(|(_, urgency)| *urgency);

                    // 依次调度，返回调度后的最大offset
                    flows_to_sched
                        .iter()
                        .map(|(flow, _)| *flow)
                        .map(|f| {
                            if no_seq {
                                f.schedule_link(flows.values(), link)
                            } else {
                                f.schedule_link(seq_flows.values(), link)
                            }
                        })
                        .max()
                        .unwrap_or(0)
                };

                // 根据队列大小选择串行或并行处理，并更新下一时序的起点
                let max_offset = if queue.len() < PARALLEL_THRESHOLD {
                    queue.iter().map(schedule_link).max().unwrap_or(0)
                } else {
                    queue.par_iter().map(schedule_link).max().unwrap_or(0)
                };

                start_offset = start_offset.max(max_offset);

                // 移除已处理的节点
                for link in &queue {
                    for node in pre_graph.node_indices() {
                        if pre_graph[node].id == link.id {
                            pre_graph.remove_node(node);
                            break;
                        }
                    }
                }
            }
        }

        if END_BREAKLOOP {
            // 按照紧急程度调度
            for (f, _) in breakloop.iter().rev() {
                for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                    start_offset = start_offset.max(
                        if no_seq {
                            f.schedule_link(flows.values(), link)
                        } else {
                            f.schedule_link(seq_flows.values(), link)
                        }
                    );
                }
            }
        }
    }

    match output_type {
        OutputType::Inet(filename) => output_inet::output(&processed_input, filename),
        OutputType::Console => {
            for (name, flow) in flows {
                println!("Flow {}", name);
                for entry in  &flow.link_offsets {
                    let (from, to) = entry.key();
                    let offset = entry.value();
                    println!("\t{} -> {}: {}", from, to, offset);
                }
            }
        }
    }
}

fn find_cycle_edges(graph: &Graph<&model::Link, Route>) -> Option<FxHashSet<EdgeIndex>> {
    let sccs = tarjan_scc(graph); // 使用 Tarjan 算法找到所有强连通分量
    let mut cycle_edges = FxHashSet::default();

    for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
        for &node_i in &scc {
            for (edge, node_j) in graph.edges(node_i).map(|e| (e.id(), e.target())) {
                if scc.contains(&node_j) {
                    cycle_edges.insert(edge);
                }
            }
        }
    }

    if cycle_edges.is_empty() {
        None
    } else {
        Some(cycle_edges)
    }
}
