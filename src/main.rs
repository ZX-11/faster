#[macro_use]
extern crate derive_deref;

use fxhash::FxHashMap;
use model::{Network, PreGraph, ProcessedInput, Route};
use petgraph::{algo::tarjan_scc, graph::NodeIndex, Graph};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use ustr::UstrMap;

mod input_inet;
mod input_json;
mod model;
mod output_inet;

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

    let start_offset = Arc::new(AtomicU64::new(0));

    for (sequence, seq_flows) in flow_sequence {
        let no_seq = sequence == u32::MAX;
        let start = if no_seq {
            0
        } else {
            start_offset.load(Ordering::Relaxed)
        };

        // 构建先序关系图
        let mut pre_graph = PreGraph::default();

        let link_id = links
            .iter()
            .map(|(id, ln)| (*id, pre_graph.add_node(ln)))
            .collect::<FxHashMap<_, _>>();

        for flow in seq_flows.values() {
            flow.start_offset.store(start, Ordering::Relaxed);

            for route @ (hop, next_hop) in &flow.routes {
                pre_graph.add_edge(link_id[hop], link_id[next_hop], Route { id: *route });
            }
        }

        let mut queue = Vec::new();
        let mut breakloop = Vec::new();

        while pre_graph.node_count() != 0 {
            // 选择入度为0的节点加入队列
            for node in pre_graph.node_indices() {
                let indegree = pre_graph
                    .neighbors_directed(node, petgraph::Incoming)
                    .count();
                if indegree == 0 {
                    queue.push(pre_graph[node]);
                }
            }

            if queue.is_empty() {
                // 循环直到图中没有环
                while let Some(cycle_edges) = find_cycle_edges(&pre_graph.graph) {
                    // 映射边到网络流量序列
                    let mut edge_to_flows = FxHashMap::default();
                    for flow in flows.values() {
                        for (hop, next_hop) in &flow.routes {
                            let edge = (link_id[hop], link_id[next_hop]);
                            edge_to_flows
                                .entry(edge)
                                .or_insert_with(Vec::new)
                                .push(flow);
                        }
                    }

                    // 统计每个网络流量序列包含的环边数量
                    let mut flow_to_cycle_edge_count = UstrMap::default();
                    for edge in &cycle_edges {
                        if let Some(flows) = edge_to_flows.get(edge) {
                            for flow in flows {
                                *flow_to_cycle_edge_count.entry(flow.id).or_insert(0) += 1;
                            }
                        }
                    }

                    // 记录被破环的流
                    if let Some((id, _)) = flow_to_cycle_edge_count
                        .iter()
                        .max_by_key(|(_, count)| *count)
                    {
                        let flow = &flows[id];

                        flow.breakloop.store(true, Ordering::Relaxed);

                        breakloop.push((
                            flow,
                            flow.calculate_urgency(flow.first_unscheduled_link())
                                .unwrap(),
                        ));

                        // 从图中移除该流的所有边
                        pre_graph.breakloop(flow);

                        // 移除孤立的节点
                        pre_graph.remove_isolated_nodes();
                    }
                }

                breakloop.sort_unstable_by_key(|(_, urgency)| *urgency);

                for (f, _) in breakloop.iter() {
                    for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                        start_offset
                            .fetch_max(f.schedule_link(flows.values(), link), Ordering::Relaxed);
                    }
                }

                // 清空破环列表，准备下一轮处理
                breakloop.clear();
            } else {
                const PARALLEL_THRESHOLD: usize = 8;

                let start_offset = start_offset.clone();

                let schedule_link = |link: &&model::Link| {
                    // 获取当前链路待调度的流
                    let mut flows_to_sched = link
                        .flows
                        .iter()
                        .map(|f| &flows[f])
                        .filter(|f| {
                            seq_flows.contains_key(&f.id)
                                && !f.schedule_done()
                                && !f.breakloop.load(Ordering::Relaxed)
                        })
                        .map(|f| (f, f.calculate_urgency(link).unwrap()))
                        .collect::<Vec<_>>();

                    // 按紧急程度排序
                    flows_to_sched.sort_unstable_by_key(|(_, urgency)| *urgency);

                    // 依次调度，并更新下一时序的起点
                    flows_to_sched.iter().map(|(flow, _)| flow).for_each(|f| {
                        start_offset.fetch_max(
                            if no_seq {
                                f.schedule_link(flows.values(), link)
                            } else {
                                f.schedule_link(seq_flows.values(), link)
                            },
                            if queue.len() < PARALLEL_THRESHOLD {
                                Ordering::Relaxed
                            } else {
                                Ordering::SeqCst
                            },
                        );
                    });
                };

                // 根据队列大小选择串行或并行处理
                if queue.len() < PARALLEL_THRESHOLD {
                    queue.iter().for_each(schedule_link);
                } else {
                    queue.par_iter().for_each(schedule_link);
                }

                // 移除已处理的节点
                for link in &queue {
                    for node in pre_graph.node_indices() {
                        if pre_graph[node].id == link.id {
                            pre_graph.remove_node(node);
                            break;
                        }
                    }
                }

                queue.clear();
            }
        }
    }

    match output_type {
        OutputType::Inet(filename) => output_inet::output(&processed_input, filename),
        OutputType::Console => {
            for (name, flow) in flows {
                println!("Flow {}", name);
                flow.link_offsets.iter().for_each(|entry| {
                    let (from, to) = entry.key();
                    let offset = entry.value();
                    println!("\t{} -> {}: {}", from, to, offset);
                });
            }
        }
    }
}

fn find_cycle_edges(graph: &Graph<&model::Link, Route>) -> Option<Vec<(NodeIndex, NodeIndex)>> {
    let sccs = tarjan_scc(graph); // 使用 Tarjan 算法找到所有强连通分量
    let mut cycle_edges = Vec::new();

    // 遍历每个强连通分量
    for scc in sccs {
        // 如果强连通分量的大小大于 1，说明它是一个环
        if scc.len() > 1 {
            // 遍历强连通分量中的节点
            for (i, &node_i) in scc.iter().enumerate() {
                for j in (i + 1)..scc.len() {
                    let node_j = unsafe { *scc.get_unchecked(j) };

                    // 检查从 node_i 到 node_j 是否存在边
                    if graph.contains_edge(node_i, node_j) {
                        cycle_edges.push((node_i, node_j));
                    }

                    // 检查从 node_j 到 node_i 是否存在边
                    if graph.contains_edge(node_j, node_i) {
                        cycle_edges.push((node_j, node_i));
                    }
                }
            }
        }
    }

    match cycle_edges.len() {
        0 => None,
        _ => Some(cycle_edges),
    }
}
