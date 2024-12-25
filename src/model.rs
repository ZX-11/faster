use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use petgraph::{prelude::{StableGraph, EdgeIndex}, Graph};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::{
    any::Any,
    collections::VecDeque,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    u64,
};
use ustr::{Ustr, UstrMap};

type FxIndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;
type FxDashMap<K, V> = dashmap::DashMap<K, V, FxBuildHasher>;

pub type FlowID = Ustr;
pub type LinkID = (Ustr, Ustr);
pub type RouteID = ((Ustr, Ustr), (Ustr, Ustr));

#[derive(Default)]
pub struct ProcessedInput {
    pub devices: UstrMap<Device>,
    pub links: FxHashMap<(Ustr, Ustr), Link>,
    pub flows: UstrMap<Flow<'static>>,
    pub addition: Option<Box<dyn Any>>,
}

static mut PROCESSED_INPUT: Option<ProcessedInput> = None;

// 这个函数只能被调用一次，用于读取输入，示例用法见input_json.rs
pub fn init_processed_input() -> &'static mut ProcessedInput {
    unsafe {
        let p = &mut *&raw mut PROCESSED_INPUT;
        assert!(p.is_none());
        *p = Some(ProcessedInput::default());
        p.as_mut().unwrap_unchecked()
    }
}

// 调用前必须已经调用init_processed_input()
pub fn processed_input() -> &'static ProcessedInput {
    unsafe {
        (&*&raw const PROCESSED_INPUT).as_ref().unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Device {
    // pub id: Ustr,
    pub pdelay: u32,
    pub end_device: bool,
}

#[derive(Debug)]
pub struct Link {
    pub id: LinkID,
    pub from: Device,
    pub _to: Device,
    pub delay: u32,
    pub speed: u32,
    pub flows: Vec<FlowID>,
}

#[allow(dead_code)]
pub struct Route {
    pub id: RouteID,
}

#[derive(Debug, Default)]
pub struct Flow<'a> {
    pub id: Ustr,
    pub length: u32,
    pub period: u64,
    pub max_latency: u64,
    pub sequence: u32,

    // 拓扑相关
    pub links: FxIndexMap<(Ustr, Ustr), &'a Link>, // 保持插入顺序的哈希表
    pub predecessors: FxHashMap<(Ustr, Ustr), Vec<(Ustr, Ustr)>>,
    pub remain_min_delay: FxHashMap<(Ustr, Ustr), u64>,
    pub routes: Vec<RouteID>,

    // 调度状态
    pub link_offsets: FxDashMap<(Ustr, Ustr), u64>,
    pub start_offset: AtomicU64,
    pub schedule_done: AtomicBool,
    pub breakloop: AtomicBool,
}

impl<'a> AsRef<Flow<'a>> for Flow<'a> {
    fn as_ref(&self) -> &Flow<'a> {
        self
    }
}

impl<'a> Flow<'a> {
    #[inline]
    pub fn schedule_done(&self) -> bool {
        self.schedule_done.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn start_offset(&self) -> u64 {
        self.start_offset.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn breakloop(&self) -> bool {
        self.breakloop.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn link_offset(&self, link_id: &(Ustr, Ustr)) -> u64 {
        *self.link_offsets.get(link_id).unwrap()
    }

    pub fn calculate_urgency(&self, link: &'a Link) -> Option<u64> {
        let total_delay =
            self.sfdelay(link) + self.accdelay(link) + self.remain_min_delay[&link.id];

        match self.max_latency as i64 - total_delay as i64 {
            x if x < 0 => None,
            x => Some(x as u64),
        }
    }

    #[inline]
    pub fn ldelay(&self, link: &'a Link) -> u64 {
        link.delay as u64
    }

    #[inline]
    pub fn pdelay(&self, link: &'a Link) -> u64 {
        link.from.pdelay as u64
    }

    #[inline]
    pub fn tdelay(&self, link: &'a Link) -> u64 {
        (self.length * 8000 / link.speed) as u64
    }

    #[inline]
    pub fn sfdelay(&self, link: &'a Link) -> u64 {
        self.pdelay(link) // 目前未考虑接收存储转发所需时间
    }

    #[inline]
    pub fn scheduled_link(&self, link: &'a Link) -> bool {
        self.link_offsets.contains_key(&link.id)
    }

    pub fn first_unscheduled_link(&self) -> &'a Link {
        self.links
            .values()
            .filter(|l| !self.scheduled_link(l))
            .next()
            .unwrap()
    }

    pub fn schedule_link(
        &self,
        flows: impl Iterator<Item = impl AsRef<Flow<'a>>>,
        link: &'a Link,
    ) -> u64 {
        let earliest = self.accdelay(link) + self.pdelay(link);

        // 获取当前链路已经完成调度的流
        let prevs: SmallVec<[_; 64]> = flows
            .filter(|f| f.as_ref().scheduled_link(link))
            .collect();

        // 计算宏周期
        let max_time = lcm(
            self.period,
            lcm_all(prevs.iter().map(|f| f.as_ref().period)),
        );

        // 计算已占用时间槽列表
        let mut occupied: SmallVec<[_; 512]> = SmallVec::from_slice(&[(max_time, max_time)]);
        let mut earliest_occupied = false;

        for prev in &prevs {
            let prev = prev.as_ref();
            for i in 0..max_time / prev.period {
                let start = i * prev.period + prev.link_offset(&link.id);
                let end = start + prev.tdelay(link);
                if start <= earliest && earliest <= end {
                    earliest_occupied = true;
                }
                occupied.push((start, end));
            }
        }

        if !earliest_occupied {
            occupied.push((earliest, earliest));
        }

        occupied.sort();

        let possible_starts: FxHashSet<_> = occupied
            .windows(2)
            .filter_map(|slots| {
                let (_, prev_end) = unsafe { *slots.get_unchecked(0) };
                let (next_start, _) = unsafe { *slots.get_unchecked(1) };
                let start = prev_end % self.period;
                match start >= earliest && next_start >= prev_end + self.tdelay(link) {
                    true => Some(start),
                    false => None,
                }
            })
            .collect();

        let found = possible_starts
            .par_iter()
            .filter_map(|&start| {
                let slots: SmallVec<[_; 128]> = (0..max_time / self.period)
                    .map(|i| i * self.period + start)
                    .map(|start| (start, start + self.tdelay(link)))
                    .collect();
                match conflict_with(&slots, &occupied) {
                    true => None,
                    false => Some(start),
                }
            })
            .min()
            .expect(&format!("No feasible schedule found for flow {}", self.id));

        // 更新调度状态
        self.link_offsets.insert(link.id, found);
        if self.link_offsets.len() == self.links.len() {
            self.schedule_done.store(true, Ordering::Relaxed);
        }

        // 返回时间槽结束offset
        found + self.tdelay(link) + self.ldelay(link)
    }

    fn accdelay(&self, link: &Link) -> u64 {
        match self.predecessors.get(&link.id) {
            Some(pred) => pred
                .iter()
                .map(|l| self.link_offset(l) + self.tdelay(self.links[l]))
                .max()
                .unwrap_or(self.start_offset()),
            None => self.start_offset(),
        }
    }

    pub fn generate_remain_min_delay(mut self, links: &FxHashMap<(Ustr, Ustr), Link>) -> Self {
        // 预构建入边映射和出度映射
        let graph = self.links.values().map(|l| l.id).collect::<Vec<_>>();
        let mut in_edges: UstrMap<SmallVec<[(Ustr, Ustr); 16]>> = Default::default();
        let mut out_degree: UstrMap<usize> = Default::default();

        // 构建入边映射和出度映射
        for &edge @ (from, to) in &graph {
            in_edges.entry(to).or_default().push(edge);
            out_degree.entry(to).or_default();
            *out_degree.entry(from).or_default() += 1;
        }

        // 初始化结果映射和待处理边集
        let mut result: FxHashMap<(Ustr, Ustr), u64> = Default::default();
        let mut remaining_edges: FxHashSet<(Ustr, Ustr)> = graph.iter().cloned().collect();

        // 找出出度为0的节点队列
        let mut zero_out_nodes: SmallVec<[Ustr; 32]> = out_degree
            .iter()
            .filter_map(|(node, &degree)| {
                if degree == 0 {
                    Some(node.clone())
                } else {
                    None
                }
            })
            .collect();

        while !zero_out_nodes.is_empty() {
            let current_zero_out_nodes = zero_out_nodes;
            zero_out_nodes = SmallVec::new();

            for node in current_zero_out_nodes {
                // 处理指向该节点的入边
                if let Some(edges_to_node) = in_edges.get(&node) {
                    for edge in edges_to_node {
                        let link = &links[edge];
                        // 计算并存储边的值
                        let prev = result
                            .iter()
                            .filter(|(link, _)| link.0 == edge.1)
                            .map(|(_, prev)| *prev)
                            .max()
                            .unwrap_or(0);
                        result.insert(
                            *edge,
                            prev + self.pdelay(&link) + self.tdelay(&link) + self.ldelay(&link),
                        );

                        // 移除该边
                        remaining_edges.remove(edge);

                        // 减少起始节点的出度
                        if let Some(degree) = out_degree.get_mut(&edge.0) {
                            *degree -= 1;

                            // 如果起始节点出度变为0，加入下一轮处理
                            if *degree == 0 {
                                zero_out_nodes.push(edge.0);
                            }
                        }
                    }
                }
            }

            if zero_out_nodes.is_empty() && !remaining_edges.is_empty() {
                zero_out_nodes = remaining_edges.iter().map(|edge| edge.0).collect();
            }
        }

        self.remain_min_delay = result;
        self
    }

    pub fn generate_routes(mut self) -> Self {
        if !self.routes.is_empty() {
            return self;
        }

        let mut end_map: UstrMap<SmallVec<[(Ustr, Ustr); 16]>> = Default::default();

        for hop in self.links.values().map(|l| l.id) {
            end_map.entry(hop.0).or_default().push(hop);
        }

        for hop in self.links.values().map(|l| l.id) {
            if let Some(next_hops) = end_map.get(&hop.1) {
                for &next_hop in next_hops {
                    assert!(next_hop != hop);
                    self.routes.push((hop, next_hop));
                }
            }
        }

        self
    }
}

impl<'a> PartialEq for Flow<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Deref, DerefMut, Default)]
pub struct Network<'a> {
    pub graph: Graph<&'a Device, &'a Link>,
}

#[derive(Deref, DerefMut, Default)]
pub struct PreGraph<'a> {
    pub graph: StableGraph<&'a Link, Route>,
}

impl<'a> PreGraph<'a> {
    /// 移除指定流的所有路由边，并标记该流为破环状态
    pub fn breakloop(&mut self, flow: &Flow, route_id: &FxHashMap<(FlowID, RouteID), EdgeIndex>) {
        // 标记流为破环状态
        flow.breakloop.store(true, Ordering::Relaxed);

        // 移除流的所有路由边
        for route in flow.routes.iter() {
            self.graph.remove_edge(route_id[&(flow.id, *route)]);
        }

        // 移除孤立的节点
        self.remove_isolated_nodes();
    }

    /// 移除所有没有连接的节点
    pub fn remove_isolated_nodes(&mut self) {
        self.graph.retain_nodes(|graph, node| {
            let in_degree = graph.neighbors_directed(node, petgraph::Incoming).count();
            let out_degree = graph.neighbors_directed(node, petgraph::Outgoing).count();
            in_degree > 0 || out_degree > 0
        });
    }
}

// 二分查找，时间复杂度为O(n log m)，保留该算法的实现供比较测试
pub fn _conflict_with(slots: &[(u64, u64)], occupied: &[(u64, u64)]) -> bool {
    if occupied.is_empty() {
        return false;
    }
    let conflict = |&(start, end)| {
        occupied
            .binary_search_by(|&(occupied_start, occupied_end)| match true {
                _ if occupied_end <= start => std::cmp::Ordering::Less,
                _ if occupied_start >= end => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            })
            .is_ok()
    };
    if slots.len() < 64 && occupied.len() < 512 {
        // 串行处理
        slots.into_iter().any(conflict)
    } else {
        // 并行处理
        slots.into_par_iter().any(conflict)
    }
}

// 使用双指针法检查冲突，时间复杂度为O(n + m)
pub fn conflict_with(slots: &[(u64, u64)], occupied: &[(u64, u64)]) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < slots.len() && j < occupied.len() {
        let (s1, e1) = unsafe { *slots.get_unchecked(i) };
        let (s2, e2) = unsafe { *occupied.get_unchecked(j) };

        if e1 <= s2 {
            i += 1;
        } else if e2 <= s1 {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

pub fn sort_hops(
    hops: &FxHashSet<(Ustr, Ustr)>,
) -> (
    Vec<(Ustr, Ustr)>,
    FxHashMap<(Ustr, Ustr), Vec<(Ustr, Ustr)>>,
) {
    // 构建图的邻接表和入度表
    let mut adjacency_list: UstrMap<Vec<Ustr>> = Default::default();
    let mut in_degree: UstrMap<usize> = Default::default();
    let mut predecessors: FxHashMap<(Ustr, Ustr), Vec<(Ustr, Ustr)>> = Default::default();

    for &(from, to) in hops {
        adjacency_list.entry(from).or_insert_with(Vec::new).push(to);
        *in_degree.entry(to).or_insert(0) += 1;
        in_degree.entry(from).or_insert(0);
    }

    // 找到所有入度为 0 的节点
    let mut queue: VecDeque<Ustr> = in_degree
        .iter()
        .filter(|(_, &degree)| degree == 0)
        .map(|(&node, _)| node)
        .collect();

    let mut sorted_hops = Vec::with_capacity(hops.len());

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = adjacency_list.get(&node) {
            for neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(neighbor) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*neighbor);
                    }
                }
            }
        }

        // 找到所有以 node 为起点的边，并加入结果列表
        sorted_hops.extend(hops.iter().filter(|(from, _)| *from == node));
    }

    // 构建 predecessors 映射
    for (from, to) in hops {
        for (pred_from, pred_to) in hops {
            if pred_to == from {
                predecessors
                    .entry((*from, *to))
                    .or_insert_with(Vec::new)
                    .push((*pred_from, *pred_to));
            }
        }
    }

    (sorted_hops, predecessors)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    (a / gcd(a, b)) * b
}

fn lcm_all(it: impl Iterator<Item = u64>) -> u64 {
    let mut acc: u64 = 1;
    for n in it {
        if n == 0 {
            return 0;
        }
        acc = (acc * n) / gcd(acc, n);
    }
    acc
}
