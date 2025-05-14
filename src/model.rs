use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use petgraph::{
    prelude::{EdgeIndex, StableGraph},
    Graph,
};
use smallvec::SmallVec;
use std::{
    any::Any,
    cell::{Cell, RefCell, UnsafeCell},
    collections::VecDeque,
    u32,
};
use ustr::{Ustr, UstrMap, UstrSet};

pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;

pub type FlowID = Ustr;
pub type LinkID = (Ustr, Ustr);
pub type RouteID = (LinkID, LinkID);

#[derive(Default)]
pub struct ProcessedInput {
    pub devices: UstrMap<Device>,
    pub links: FxHashMap<LinkID, Link>,
    pub flows: UstrMap<Flow<'static>>,
    pub addition: Option<Box<dyn Any>>,
}

static mut PROCESSED_INPUT: Option<ProcessedInput> = None;

// 这个函数只能被调用一次，用于读取输入，示例用法见input_json.rs
pub fn init_processed_input() -> &'static mut ProcessedInput {
    unsafe {
        let p = &mut *std::ptr::addr_of_mut!(PROCESSED_INPUT);
        assert!(p.is_none());
        *p = Some(ProcessedInput::default());
        p.as_mut().unwrap_unchecked()
    }
}

// 调用前必须已经调用init_processed_input()
pub fn processed_input() -> &'static ProcessedInput {
    unsafe { (&*std::ptr::addr_of!(PROCESSED_INPUT)).as_ref().unwrap() }
}

macro_rules! new {
    () => {
        Default::default()
    };
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Device {
    // pub id: Ustr,
    pub pdelay: u32,
    pub end_device: bool,
}

#[derive(Debug, Default)]
pub struct Link {
    pub id: LinkID,
    pub from: Device,
    pub _to: Device,
    pub delay: u32,
    pub speed: u32,
    pub flows: Vec<FlowID>,

    pub occupied: UnsafeCell<FxIndexMap<u32, Vec<(u64, u64)>>>,
    pub macro_time: RefCell<FxHashMap<u32, u64>>,
    pub max_time: RefCell<FxHashMap<u32, u64>>,
}

impl Link {
    #[inline]
    fn occupied(&self) -> &mut FxIndexMap<u32, Vec<(u64, u64)>> {
        unsafe { self.occupied.get().as_mut().unwrap_unchecked() }
    }

    fn merge_slots(&self, slots: &[(u64, u64)], sequence: u32, min_gap: u64) {
        let occupied = self.occupied().entry(sequence).or_default();

        if occupied.is_empty() {
            *occupied = slots.to_vec();
            return;
        }

        let mut new: Vec<(u64, u64)> = Vec::with_capacity(occupied.len() + slots.len());
        let mut i = 0;
        let mut j = 0;

        while i < occupied.len() && j < slots.len() {
            unsafe {
                match (
                    occupied.get_unchecked(i),
                    slots.get_unchecked(j),
                    new.last_mut(),
                ) {
                    (x, _, Some(last)) if last.1 + min_gap >= x.0 => {
                        last.1 = x.1;
                        i += 1;
                    }
                    (_, y, Some(last)) if last.1 + min_gap >= y.0 => {
                        last.1 = y.1;
                        j += 1;
                    }
                    (x, y, _) if x.0 < y.0 => {
                        new.push(*x);
                        i += 1;
                    }
                    (_, y, _) => {
                        new.push(*y);
                        j += 1;
                    }
                }
            }
        }

        let mut remaining = |idx: usize, list: &[(u64, u64)]| {
            if idx < list.len() {
                let last = new.last_mut().unwrap();

                if list[idx].0 == last.1 {
                    last.1 = list[idx].1;
                    new.extend_from_slice(&list[idx + 1..]);
                } else {
                    new.extend_from_slice(&list[idx..]);
                }
            }
        };

        remaining(i, occupied);
        remaining(j, slots);

        *occupied = new;
    }
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
    pub links: FxIndexMap<LinkID, &'a Link>, // 保持插入顺序的哈希表
    pub predecessors: FxHashMap<LinkID, Vec<LinkID>>,
    pub remain_min_delay: FxHashMap<LinkID, u64>,
    pub routes: Vec<RouteID>,

    // 调度状态
    pub link_offsets: UnsafeCell<FxHashMap<LinkID, u64>>,
    pub start_offset: Cell<u64>,
    pub schedule_done: Cell<bool>,
    pub schedule_fail: Cell<bool>,
    pub breakloop: Cell<bool>,
}

impl<'a> AsRef<Flow<'a>> for Flow<'a> {
    fn as_ref(&self) -> &Flow<'a> {
        self
    }
}

impl<'a> Flow<'a> {
    #[inline]
    pub fn offset_of(&self, link_id: &LinkID) -> u64 {
        self.link_offsets()[link_id]
    }

    #[inline]
    pub fn link_offsets(&self) -> &mut FxHashMap<LinkID, u64> {
        unsafe { self.link_offsets.get().as_mut().unwrap_unchecked() }
    }

    pub fn calculate_urgency(&self, link: &'a Link) -> Option<u64> {
        let total_delay =
            self.sfdelay(link) + self.accdelay(link) + self.remain_min_delay[&link.id]
                - self.start_offset.get();

        match self.max_latency as i64 - total_delay as i64 {
            x if x < 0 => {
                self.schedule_fail.set(true);
                None
            }
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
        (match link.speed {
            10 => self.length * 800,
            100 => self.length * 80,
            1000 => self.length * 8,
            _ => self.length * 8000 / link.speed,
        }) as u64
    }

    #[inline]
    pub fn sfdelay(&self, link: &'a Link) -> u64 {
        self.pdelay(link) // 目前未考虑接收存储转发所需时间
    }

    #[inline]
    pub fn scheduled_link(&self, link: &'a Link) -> bool {
        self.link_offsets().contains_key(&link.id)
    }

    #[inline]
    pub fn scheduled_link_id(&self, link: &LinkID) -> bool {
        self.link_offsets().contains_key(link)
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
        sequence: u32,
    ) -> Option<u64> {
        let earliest = self.accdelay(link) + self.pdelay(link);

        // 计算已调度最大周期
        let max_time = {
            let mut m = link.max_time.borrow_mut();
            let max_time = *m
                .entry(sequence)
                .and_modify(|v| *v = lcm(self.period, *v))
                .or_insert(self.period);
            if sequence != u32::MAX {
                m.entry(u32::MAX)
                    .and_modify(|v| *v = lcm(*v, max_time))
                    .or_insert(max_time);
            }
            max_time
        };

        // 计算宏周期
        let macro_time = link
            .macro_time
            .borrow_mut()
            .entry(sequence)
            .or_insert(lcm_all(flows.map(|f| f.as_ref().period)))
            .clone();

        // 初始化无时序已分配时间槽
        if sequence == u32::MAX && !link.occupied().contains_key(&u32::MAX) {
            link.occupied().insert(
                u32::MAX,
                link.occupied()
                    .values()
                    .flat_map(|i| i.iter())
                    .cloned()
                    .collect(),
            );
        }

        // 取出与本次调度相关的时间槽
        let occupied = link.occupied().entry(sequence).or_default();

        let start = occupied
            .binary_search_by_key(&earliest, |slot| slot.1)
            .unwrap_or_else(|i| i);
        let end = occupied[start..]
            .binary_search_by_key(&max_time, |slot| slot.0)
            .unwrap_or_else(|i| i);

        let occupied = &occupied[start..start + end];

        let tdelay = self.tdelay(link);

        let min_gap = 512000 / link.speed;

        // 找到所有待检测解并排序去重
        let mut possible_starts: SmallVec<[_; 512]> = occupied
            .iter()
            .zip(occupied.iter().skip(1))
            .filter_map(|(&(_, prev_end), &(next_start, _))| {
                let start = _mod(prev_end, self.period);
                match start >= earliest && next_start >= prev_end + tdelay {
                    true => Some(start),
                    false => None,
                }
            })
            .collect();

        possible_starts.push(earliest);

        if let Some(last) = occupied.last() {
            if _mod(last.1, self.period) + tdelay <= self.period {
                possible_starts.push(last.1);
            }
        }

        // 此处使用基数排序，时间复杂度为O(n)
        radsort::sort(&mut possible_starts);
        possible_starts.dedup();

        // 找到最小可行解
        match possible_starts.iter().find(|&start| {
            let n = (max_time / self.period) as usize;
            let slots: SmallVec<[_; 128]> = (0..macro_time / self.period)
                .map(|i| i * self.period + start)
                .map(|start| (start, start + tdelay))
                .collect();

            let ok = !conflict_with(&slots[..n], &occupied);
            // 将找到的最小可行解归并到已占用时间槽列表中
            if ok {
                link.merge_slots(&slots, sequence, min_gap as u64);
            }
            ok
        }) {
            Some(&found) => {
                // 更新调度状态
                let link_offsets = self.link_offsets();
                link_offsets.insert(link.id, found);
                if link_offsets.len() == self.links.len() {
                    self.schedule_done.set(true);
                }

                // 返回时间槽结束offset
                Some(found + tdelay + self.ldelay(link))
            }
            None => {
                self.schedule_fail.set(true);
                None
            }
        }
    }

    fn accdelay(&self, link: &Link) -> u64 {
        match self.predecessors.get(&link.id) {
            Some(pred) => pred
                .iter()
                .map(|l| self.offset_of(l) + self.tdelay(self.links[l]))
                .max()
                .unwrap_or(self.start_offset.get()),
            None => self.start_offset.get(),
        }
    }

    pub fn generate_remain_min_delay_p2p(mut self) -> Self {
        let mut result: FxHashMap<LinkID, u64> = new!();
        let mut acc = 0;
        for link in self.links.values().rev() {
            result.insert(link.id, acc);
            acc += self.pdelay(link) + self.tdelay(link) + self.ldelay(link);
        }
        self.remain_min_delay = result;
        self
    }

    pub fn generate_remain_min_delay(mut self, links: &FxHashMap<LinkID, Link>) -> Self {
        // 1. 利用所有链路（links.keys()）计算终端设备集合
        let end_nodes_set = end_nodes(self.links.keys());

        // 2. 统计每个 pred 链路的后继数量
        let mut remaining_successors: FxHashMap<LinkID, u32> = new!();
        for &(pred, _succ) in &self.routes {
            *remaining_successors.entry(pred).or_insert(0) += 1;
        }

        // 3. 初始化结果映射：对于那些在 route_links 中，其目标设备 (to) 在 end_nodes_set 中的链路，
        //    直接计算 remain_min_delay = pdelay + tdelay + ldelay
        let mut result = FxHashMap::default();
        let mut queue = VecDeque::new();

        for id @ (_, to) in self.links.keys() {
            if end_nodes_set.contains(to) {
                let link = &links[id];
                let delay = self.pdelay(link) + self.tdelay(link) + self.ldelay(link);
                result.insert(*id, delay);
                queue.push_back(*id);
            }
        }

        // 4. 反向传播：当队列非空时，依次处理每个链路，更新其所有前驱的 remain_min_delay
        while let Some(curr) = queue.pop_front() {
            if let Some(preds) = self.predecessors.get(&curr) {
                for &pred in preds {
                    // 对 pred 的计数做递减
                    let count = remaining_successors.get_mut(&pred).unwrap();
                    *count -= 1;
                    // 计算当前候选值
                    let pred_link = &links[&pred];
                    let candidate = self.pdelay(pred_link)
                        + self.tdelay(pred_link)
                        + self.ldelay(pred_link)
                        + result[&curr];
                    result
                        .entry(pred)
                        .and_modify(|v| {
                            if candidate > *v {
                                *v = candidate;
                            }
                        })
                        .or_insert(candidate);
                    // 当 pred 的所有后继均处理完毕，加入队列
                    if *count == 0 {
                        queue.push_back(pred);
                    }
                }
            }
        }

        self.remain_min_delay = result;
        self
    }

    pub fn generate_routes(mut self) -> Self {
        if !self.routes.is_empty() {
            return self;
        }

        let mut end_map: UstrMap<Vec<LinkID>> = new!();

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

fn end_nodes<'a>(links: impl Iterator<Item = &'a LinkID>) -> UstrSet {
    let mut neighbors: UstrMap<UstrSet> = new!();
    let mut indegree: UstrMap<u32> = new!();

    for &(from, to) in links {
        neighbors.entry(from).or_default().insert(to);
        neighbors.entry(to).or_default().insert(from);
        *indegree.entry(to).or_default() += 1;
    }

    neighbors
        .into_iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(k, _)| k)
        .filter(|k| indegree.get(k) == Some(&1))
        .collect()
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
        flow.breakloop.set(true);

        // 移除流的所有路由边
        for route in flow.routes.iter() {
            // self.graph.remove_edge(route_id[&(flow.id, *route)]);
            if let Some(edge) = route_id.get(&(flow.id, *route)) {
                self.graph.remove_edge(*edge);
            }
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

// 结合双指针和二分查找，时间复杂度为O(n * log m)
fn conflict_with(slots: &[(u64, u64)], occupied: &[(u64, u64)]) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < slots.len() && j < occupied.len() {
        let (s1, e1) = unsafe { *slots.get_unchecked(i) };
        let (s2, e2) = unsafe { *occupied.get_unchecked(j) };

        if e1 <= s2 {
            i += 1;
        } else if e2 <= s1 {
            if j + 1 >= occupied.len() {
                return false;
            }
            // 使用二分查找找到第一个 e2 > s1 的项
            j += 1 + match occupied[j + 1..].binary_search_by_key(&s1, |&(_, e)| e) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
        } else {
            return true;
        }
    }
    false
}

pub fn sort_hops(hops: &FxHashSet<LinkID>) -> (Vec<LinkID>, FxHashMap<LinkID, Vec<LinkID>>) {
    // 构建图的邻接表和入度表
    let mut adjacency_list: UstrMap<Vec<Ustr>> = new!();
    let mut in_degree: UstrMap<usize> = new!();
    let mut predecessors: FxHashMap<LinkID, Vec<LinkID>> = new!();

    for &(from, to) in hops {
        adjacency_list.entry(from).or_default().push(to);
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
                    .or_default()
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

#[inline]
fn lcm(a: u64, b: u64) -> u64 {
    if a == b {
        a
    } else {
        (a / gcd(a, b)) * b
    }
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

#[inline]
fn _mod(a: u64, b: u64) -> u64 {
    if a < b {
        a
    } else {
        a % b
    }
}
