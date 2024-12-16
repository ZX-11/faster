#[macro_use]
extern crate derive_deref;

use std::{collections::BTreeMap, error::Error, sync::{atomic::{AtomicU64, Ordering}, Arc}};
use petgraph::algo::is_cyclic_directed;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use ustr::UstrMap;
use fxhash::FxHashMap;
use model::{Network, PreGraph, ProcessedInput, Route};

mod input_json;
mod input_inet;
mod output_inet;
mod model;

enum InputType<'a> {
    Inet(&'a str, &'a str),
    Json(&'a str),
}
enum OutputType<'a> {
    Inet(&'a str),
    Console,
}

fn parse_args<'a>(mut args: impl Iterator<Item = &'a str>) -> Result<(InputType<'a>, OutputType<'a>), Box<dyn Error>> {
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
    let args: SmallVec::<[String; 8]> = std::env::args().collect();
    
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
        let start = if no_seq { 0 } else { start_offset.load(Ordering::Relaxed) };

        // 构建先序关系图
        let mut pre_graph = PreGraph::default();

        let link_id = links
            .iter()
            .map(|(id, ln)| (*id, pre_graph.add_node(ln)))
            .collect::<FxHashMap<_, _>>();

        for flow in seq_flows.values() {
            flow.schedule().borrow_mut().start_offset = start;

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
                while is_cyclic_directed(&pre_graph.graph) {
                    // 选择紧急度最低的流进行破环
                    let blp = seq_flows
                        .values()
                        .filter(|flow| !flow.schedule_done())
                        .min_by_key(|flow| {
                            flow.calculate_urgency(flow.first_unscheduled_link())
                                .unwrap()
                        })
                        .unwrap();

                    // 记录被破环的流
                    breakloop.push(blp);

                    // 从图中移除该流的所有边
                    pre_graph.breakloop(blp);

                    // 移除孤立的节点
                    pre_graph.remove_isolated_nodes();
                }

                // 反向处理被破环的流（从最后破环的开始）
                for f in breakloop.iter().rev() {
                    for link in f.links.iter().filter(|l| !f.scheduled_link(l)) {
                        start_offset.fetch_max(if no_seq {
                            f.schedule_link(flows.values(), link)
                        } else {
                            f.schedule_link(seq_flows.values(), link)
                        }, Ordering::SeqCst);
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
                        .filter(|f| seq_flows.contains_key(&f.id))
                        .collect::<Vec<_>>();

                    // 按紧急程度排序
                    flows_to_sched.sort_unstable_by(|a, b| {
                        let a_urgency = a.calculate_urgency(link);
                        let b_urgency = b.calculate_urgency(link);
                        a_urgency.cmp(&b_urgency)
                    });

                    // 依次调度，并更新下一时序的起点
                    flows_to_sched
                        .iter()
                        .filter(|f| !f.schedule_done())
                        .for_each(|f| {
                            start_offset.fetch_max(
                                if no_seq {
                                    f.schedule_link(flows.values(), link)
                                } else {
                                    f.schedule_link(seq_flows.values(), link)
                                },
                                if queue.len() < PARALLEL_THRESHOLD { Ordering::Relaxed } else { Ordering::SeqCst }
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
                for ((from, to), offset) in &flow.schedule().borrow().link_offsets {
                    println!("\t{} -> {}: {}", from, to, offset);
                }
            }
        }
    }
}
