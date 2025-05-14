#[macro_use]
extern crate derive_deref;

use fxhash::{FxHashMap, FxHashSet};
use input::*;
use model::{
    processed_input, Flow, FlowID, LinkID, Network, PreGraph, ProcessedInput, Route, RouteID,
};
use petgraph::{
    algo::tarjan_scc,
    graph::{EdgeIndex, NodeIndex},
    prelude::StableGraph,
    Direction::Outgoing,
};
use smallvec::SmallVec;
use std::{collections::BTreeMap, process::exit};
use ustr::{UstrMap, UstrSet};

mod input;
mod model;
mod output_inet;

const END_BREAKLOOP: bool = true;

fn no_seq(sequence: u32) -> bool {
    sequence == u32::MAX
}

fn main() {
    let args: SmallVec<[String; 8]> = std::env::args().collect();

    let (input_type, output_type) = parse_args(args.iter().map(AsRef::as_ref).skip(1))
        .unwrap_or_else(|e| {
            eprintln!("Arguement error:\n\t{}\nUse --help for more information", e);
            std::process::exit(-1);
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
            flow.start_offset
                .set(if no_seq(sequence) { 0 } else { start_offset });

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

        // let mut breakloop = Vec::new();
        let mut breakloop_flows = UstrSet::default();

        schedule_pre_graph(
            Context {
                sequence,
                seq_flows: &seq_flows,
                link_id: &link_id,
                route_id: &route_id,
                route_flows: &route_flows,
            },
            &mut start_offset,
            pre_graph,
            &mut breakloop_flows,
        );

        let mut breakloop: Vec<_> = breakloop_flows
            .into_iter()
            .filter_map(|id| {
                let flow = &flows[&id];
                flow.calculate_urgency(flow.first_unscheduled_link()).map(|urgency| (flow, urgency))
            })
            .collect();

        breakloop.sort_by_key(|(_, u)| *u);

        if END_BREAKLOOP {
            // 按照紧急程度调度
            for (f, _) in breakloop.iter() {
                for link in f.links.values().filter(|l| !f.scheduled_link(l)) {
                    match match no_seq(sequence) {
                        true => f.schedule_link(flows.values(), link, sequence),
                        false => f.schedule_link(seq_flows.values(), link, sequence),
                    } {
                        Some(result) => start_offset = start_offset.max(result),
                        None => break,
                    }
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

    let failed_count = flows.values().filter(|f| f.schedule_fail.get()).count();
    if failed_count != 0 {
        eprintln!("{}", failed_count);
        exit(failed_count as i32);
    }
}

struct Context<'a> {
    sequence: u32,
    seq_flows: &'a UstrMap<&'static Flow<'static>>,
    link_id: &'a FxHashMap<LinkID, NodeIndex>,
    route_id: &'a FxHashMap<(FlowID, RouteID), EdgeIndex>,
    route_flows: &'a FxHashMap<RouteID, Vec<&'static Flow<'static>>>,
}

fn schedule_pre_graph(
    ctx: Context<'_>,
    start_offset: &mut u64,
    mut pre_graph: PreGraph<'_>,
    breakloop_flows: &mut UstrSet,
) {
    let ProcessedInput {
        // devices,
        links,
        flows,
        ..
    } = processed_input();

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
            for blp in breakloop_points(&*pre_graph) {
                let mut breakloop_flows_inner = UstrSet::default();

                let breakloop = ctx.route_flows[&blp]
                    .iter()
                    .filter(|f| !f.schedule_done.get())
                    .filter(|f| breakloop_flows.insert(f.id) || !END_BREAKLOOP)
                    .inspect(|f| {
                        breakloop_flows_inner.insert(f.id);
                    });

                if END_BREAKLOOP {
                    for flow in breakloop {
                        pre_graph.breakloop(flow, &ctx.route_id);
                    }
                } else {
                    let breakloop: SmallVec<[_; 16]> = breakloop.collect();

                    let mut pre_graph_inner = PreGraph::default();

                    let mut link_id_inner = FxHashMap::default();

                    let mut route_id_inner = FxHashMap::default();

                    let mut route_flows_inner = FxHashMap::default();

                    for flow in &breakloop {
                        for route @ (from, to) in &flow.routes {
                            if !flow.scheduled_link_id(from) {
                                link_id_inner
                                    .entry(*from)
                                    .or_insert_with(|| pre_graph_inner.add_node(&links[from]));
                                link_id_inner
                                    .entry(*to)
                                    .or_insert_with(|| pre_graph_inner.add_node(&links[to]));

                                route_id_inner.insert(
                                    (flow.id, *route),
                                    pre_graph_inner.add_edge(
                                        link_id_inner[from],
                                        link_id_inner[to],
                                        Route { id: *route },
                                    ),
                                );

                                route_flows_inner
                                    .entry(*route)
                                    .or_insert_with(Vec::new)
                                    .push(**flow);
                            }
                        }
                        pre_graph.breakloop(flow, &ctx.route_id);
                    }

                    schedule_pre_graph(
                        Context {
                            link_id: &link_id_inner,
                            route_id: &route_id_inner,
                            route_flows: &route_flows_inner,
                            seq_flows: &breakloop.iter().map(|f| (f.id, **f)).collect(),
                            ..ctx
                        },
                        start_offset,
                        pre_graph_inner,
                        &mut breakloop_flows_inner,
                    );

                    for flow in &breakloop {
                        breakloop_flows.remove(&flow.id);
                    }
                }
            }
        } else {
            let schedule_link = |link: &&model::Link| {
                // 获取当前链路待调度的流
                let mut flows_to_sched: SmallVec<[_; 64]> = link
                    .flows
                    .iter()
                    .map(|f| &flows[f])
                    .filter(|f| match (breakloop_flows.is_empty(), END_BREAKLOOP) {
                        (false, false) => {
                            ctx.seq_flows.contains_key(&f.id) && !f.schedule_done.get() && !f.schedule_fail.get()
                        }
                        _ => {
                            ctx.seq_flows.contains_key(&f.id)
                                && !f.schedule_done.get()
                                && !f.breakloop.get()
                                && !f.schedule_fail.get()
                        }
                    })
                    .filter_map(|f|
                        f.calculate_urgency(link).map(|urgency| (f, urgency))
                    )
                    .collect();

                // 按紧急程度排序
                radsort::sort_by_key(&mut flows_to_sched, |(_, urgency)| *urgency);

                // 依次调度，返回调度后的最大offset
                flows_to_sched
                    .iter()
                    .map(|(flow, _)| *flow)
                    .filter_map(|f| {
                        if no_seq(ctx.sequence) {
                            f.schedule_link(flows.values(), link, ctx.sequence)
                        } else {
                            f.schedule_link(ctx.seq_flows.values(), link, ctx.sequence)
                        }
                    })
                    .max()
                    .unwrap_or(0)
            };

            *start_offset =
                *start_offset.max(&mut queue.iter().map(schedule_link).max().unwrap_or(0));

            // 移除已处理的节点
            for link in &queue {
                pre_graph.remove_node(ctx.link_id[&link.id]);
            }
        }
    }
}

fn breakloop_points(graph: &StableGraph<&model::Link, Route>) -> Vec<RouteID> {
    let sccs = tarjan_scc(graph); // 使用 Tarjan 算法找到所有强连通分量

    let mut results = Vec::new();

    for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
        let scc = FxHashSet::from_iter(scc);
        // 收集所有边及其权重（平行边数量）
        let mut edges = FxHashMap::default();
        for &node_i in &scc {
            for node_j in graph.neighbors_directed(node_i, Outgoing) {
                if scc.contains(&node_j) {
                    let edge = if node_i < node_j {
                        (node_i, node_j)
                    } else {
                        (node_j, node_i)
                    };
                    *edges.entry(edge).or_insert(0) += 1;
                }
            }
        }

        let mut edges: Vec<_> = edges.into_iter().collect();

        // 按权重排序
        radsort::sort_by_key(&mut edges, |(_, w)| -w);

        // Kruskal计算最大生成树
        let mut uf = petgraph::unionfind::UnionFind::new(graph.capacity().0);
        let mut mst_edges = FxHashSet::default();

        for ((u, v), _) in &edges {
            if uf.union(u.index(), v.index()) {
                mst_edges.insert((u, v));
            }
        }

        // 将非最大生成树边加入结果
        for ((u, v), _) in &edges {
            if !mst_edges.contains(&(u, v)) && !mst_edges.contains(&(v, u)) {
                if graph[*u].id.1 == graph[*v].id.0 {
                    results.push((graph[*u].id, graph[*v].id));
                } else if graph[*u].id.0 == graph[*v].id.1 {
                    results.push((graph[*v].id, graph[*u].id));
                } else {
                    unreachable!();
                }
            }
        }
    }

    results
}
