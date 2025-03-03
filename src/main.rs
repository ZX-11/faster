#[macro_use]
extern crate derive_deref;

use fxhash::{FxHashMap, FxHashSet};
use input::*;
use model::{Network, PreGraph, ProcessedInput, Route};
use petgraph::{
    algo::{is_cyclic_directed, tarjan_scc},
    prelude::StableGraph,
    Direction::Outgoing,
};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use ustr::UstrMap;

mod input;
mod model;
mod output_inet;

const END_BREAKLOOP: bool = false;

fn main() {
    let args: SmallVec<[String; 8]> = std::env::args().collect();

    let (input_type, output_type) = parse_args(args.iter().map(AsRef::as_ref).skip(1))
        .unwrap_or_else(|e| {
            eprintln!("Arguement error:\n\t{}\nUse --help for more information", e);
            std::process::exit(1);
        });

    let ProcessedInput {
        devices,
        links,
        flows,
        ..
    } = process_input!(input_type => { inet, json, fast });

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

        let mut route_id = FxHashMap::default();
        let mut route_flows = FxHashMap::default();

        for flow in seq_flows.values() {
            // 为流设置最早起始时间
            flow.start_offset.set(if no_seq { 0 } else { start_offset });

            for route @ (hop, next_hop) in &flow.routes {
                route_id.insert(
                    (flow.id, *route),
                    pre_graph.add_edge(link_id[hop], link_id[next_hop], Route { id: *route }),
                );

                // 映射路由到网络流量序列
                route_flows
                    .entry(*route)
                    .or_insert_with(Vec::new)
                    .push(*flow);
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
                if END_BREAKLOOP {
                    let mut urgency = Vec::with_capacity(seq_flows.len());
                    urgency.extend(seq_flows.values().filter(|flow| !flow.schedule_done.get()).map(
                        |&flow| {
                            (
                                flow,
                                flow.calculate_urgency(flow.first_unscheduled_link())
                                    .unwrap(),
                            )
                        },
                    ));
                    urgency.sort_unstable_by_key(|(_, urgency)| *urgency);

                    while is_cyclic_directed(&pre_graph.graph) {
                        // 选择紧急度最低的流进行破环
                        let blp @ (flow, _) = urgency.pop().unwrap();

                        // 记录被破环的流
                        breakloop.push(blp);

                        // 执行破环
                        pre_graph.breakloop(flow, &route_id);
                    }
                } else {
                    // 循环直到图中没有环
                    while let Some(cycle_edges) = find_cycle_edges(&pre_graph.graph) {
                        // 找到所有造成环的边，统计每个流包含的环边数量
                        let mut flow_to_cycle_edge_count = UstrMap::default();
                        flow_to_cycle_edge_count.reserve(seq_flows.len());

                        for edge in &cycle_edges {
                            for flow in route_flows[edge].iter().filter(|f| !f.breakloop.get()) {
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
                        pre_graph.breakloop(flow, &route_id);
                    }

                    breakloop.sort_unstable_by_key(|(_, urgency)| *urgency);

                    for (f, _) in breakloop.iter() {
                        for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                            start_offset = start_offset.max(f.schedule_link(flows.values(), link, sequence));
                        }
                    }

                    // 清空破环列表，准备下一轮处理
                    breakloop.clear();
                }
            } else {
                let schedule_link = |link: &&model::Link| {
                    // 获取当前链路待调度的流
                    let mut flows_to_sched: SmallVec<[_; 64]> = link
                        .flows
                        .iter()
                        .map(|f| &flows[f])
                        .filter(|f| {
                            seq_flows.contains_key(&f.id) && !f.schedule_done.get() && !f.breakloop.get()
                        })
                        .map(|f| (f, f.calculate_urgency(link).unwrap()))
                        .collect();

                    // 按紧急程度排序
                    flows_to_sched.sort_by_key(|(_, urgency)| *urgency);

                    // 依次调度，返回调度后的最大offset
                    flows_to_sched
                        .iter()
                        .map(|(flow, _)| *flow)
                        .map(|f| {
                            if no_seq {
                                f.schedule_link(flows.values(), link, sequence)
                            } else {
                                f.schedule_link(seq_flows.values(), link, sequence)
                            }
                        })
                        .max()
                        .unwrap_or(0)
                };

                start_offset = start_offset.max(queue.iter().map(schedule_link).max().unwrap_or(0));

                // 移除已处理的节点
                for link in &queue {
                    pre_graph.remove_node(link_id[&link.id]);
                }
            }
        }

        if END_BREAKLOOP {
            // 按照紧急程度调度
            for (f, _) in breakloop.iter().rev() {
                for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                    start_offset = start_offset.max(if no_seq {
                        f.schedule_link(flows.values(), link, sequence)
                    } else {
                        f.schedule_link(seq_flows.values(), link, sequence)
                    });
                }
            }
        }
    }

    match output_type {
        OutputType::Inet(filename) => output_inet::output(filename),
        OutputType::Console => {
            for (name, flow) in flows {
                println!("Flow {}", name);
                for ((from, to), offset) in flow.link_offsets().iter() {
                    println!("\t{} -> {}: {}", from, to, offset);
                }
            }
        }
    }
}

fn find_cycle_edges(graph: &StableGraph<&model::Link, Route>) -> Option<FxHashSet<model::RouteID>> {
    let sccs = tarjan_scc(graph); // 使用 Tarjan 算法找到所有强连通分量
    let mut cycle_edges = FxHashSet::default();

    for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
        for &node_i in &scc {
            for node_j in graph.neighbors_directed(node_i, Outgoing) {
                if scc.contains(&node_j) {
                    cycle_edges.insert((graph[node_i].id, graph[node_j].id));
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
