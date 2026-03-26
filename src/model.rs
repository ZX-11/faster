use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use petgraph::{
    prelude::{EdgeIndex, StableGraph},
    Graph,
};
use std::{
    any::Any,
    cell::{Cell, UnsafeCell},
    collections::VecDeque,
    u32,
};
use ustr::{Ustr, UstrMap};

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

pub fn init_processed_input() -> &'static mut ProcessedInput {
    unsafe {
        let p = &mut *std::ptr::addr_of_mut!(PROCESSED_INPUT);
        *p = Some(ProcessedInput::default());
        p.as_mut().unwrap_unchecked()
    }
}

#[inline]
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
    pub min_gap: Cell<u32>,
    pub flows: Vec<FlowID>,

    pub occupied: UnsafeCell<Vec<(u64, u64)>>,
    pub hyper_period: Cell<u64>,
    pub max_time: Cell<u64>,
}

impl Link {
    #[inline]
    fn occupied(&self) -> &mut Vec<(u64, u64)> {
        unsafe { self.occupied.get().as_mut().unwrap_unchecked() }
    }

    #[inline]
    fn min_gap(&self) -> u64 {
        match self.min_gap.get() {
            0 => {
                let gap = 512000 / self.speed;
                self.min_gap.set(gap);
                gap as u64
            }
            x => x as u64,
        }
    }

    pub fn hyper_period(&self) -> u64 {
        let ref flows = processed_input().flows;
        match self.hyper_period.get() {
            0 => {
                let hyper_period = lcm_all(self.flows.iter().map(|f| flows[f].period));
                self.hyper_period.set(hyper_period);
                hyper_period
            }
            x => x,
        }
    }

    fn merge_slots(&self, slots: &[(u64, u64)]) {
        let occupied = self.occupied();

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
                    (x, _, Some(last)) if last.1 + self.min_gap() >= x.0 => {
                        last.1 = last.1.max(x.1);
                        i += 1;
                    }
                    (_, y, Some(last)) if last.1 + self.min_gap() >= y.0 => {
                        last.1 = last.1.max(y.1);
                        j += 1;
                    }
                    (x, y, _) => {
                        if x.0 <= y.1 + self.min_gap() && y.0 <= x.1 + self.min_gap() {
                            let start = x.0.min(y.0);
                            let end = x.1.max(y.1);

                            if let Some(last) = new.last_mut() {
                                if last.1 + self.min_gap() >= start {
                                    last.1 = last.1.max(end);
                                } else {
                                    new.push((start, end));
                                }
                            } else {
                                new.push((start, end));
                            }
                            i += 1;
                            j += 1;
                        } else if x.0 < y.0 {
                            new.push(*x);
                            i += 1;
                        } else {
                            new.push(*y);
                            j += 1;
                        }
                    }
                }
            }
        }

        let mut remaining = |idx: usize, list: &[(u64, u64)]| {
            let mut current_idx = idx;
            while current_idx < list.len() {
                if let Some(last) = new.last_mut() {
                    if list[current_idx].0 <= last.1 + self.min_gap() {
                        last.1 = last.1.max(list[current_idx].1);
                    } else {
                        new.push(list[current_idx]);
                    }
                } else {
                    new.push(list[current_idx]);
                }
                current_idx += 1;
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

    pub links: FxIndexMap<LinkID, &'a Link>,
    pub predecessors: FxHashMap<LinkID, Vec<LinkID>>,
    pub remain_min_delay: FxHashMap<LinkID, u64>,
    pub routes: Vec<RouteID>,

    pub link_offsets: UnsafeCell<FxHashMap<LinkID, u64>>,
    pub link_accdelays: UnsafeCell<FxHashMap<LinkID, u64>>,
    pub arrival: UnsafeCell<UstrMap<u64>>,

    pub pcp: Cell<i8>,
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
        if let Some(v) = self.link_offsets().get(link_id) {
            *v
        } else {
            panic!(
                "offset not found for flow {} on ({}->{})",
                self.id, link_id.0, link_id.1
            );
        }
    }

    #[inline]
    pub fn link_offsets(&self) -> &mut FxHashMap<LinkID, u64> {
        unsafe { self.link_offsets.get().as_mut().unwrap_unchecked() }
    }

    #[inline]
    pub fn arrival(&self) -> &mut UstrMap<u64> {
        unsafe { self.arrival.get().as_mut().unwrap_unchecked() }
    }

    #[inline]
    pub fn accdelay_of(&self, link_id: &LinkID) -> u64 {
        if let Some(v) = self.link_accdelays().get(link_id) {
            *v
        } else {
            panic!(
                "accdelay not found for flow {} on ({}->{})",
                self.id, link_id.0, link_id.1
            )
        }
    }

    #[inline]
    pub fn link_accdelays(&self) -> &mut FxHashMap<LinkID, u64> {
        unsafe { self.link_accdelays.get().as_mut().unwrap_unchecked() }
    }

    pub fn calculate_urgency(&self, link: &'a Link) -> Option<u64> {
        let total_delay =
            self.sfdelay(link) + self.accdelay(link) + self.remain_min_delay[&link.id];

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
        self.pdelay(link)
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
        _flows: impl Iterator<Item = impl AsRef<Flow<'a>>>,
        link: &'a Link,
        _sequence: u32,
        fifo: bool
    ) -> Option<u64> {
        let accdelay = self.accdelay(link);
        let earliest = self.start_offset.get() + accdelay + self.pdelay(link);
        
        let earliest_offset = earliest % self.period;

        let max_time = if link.max_time.get() != link.hyper_period() {
            link.max_time.update(|v| match v {
                0 => self.period,
                v => lcm(v, self.period),
            });
            link.max_time.get()
        } else {
            link.hyper_period()
        };

        let hyper_period = link.hyper_period();

        let occupied = link.occupied().as_slice(); //.entry(sequence).or_default();

        let end = match occupied.binary_search_by_key(&max_time, |slot| slot.1) {
            Ok(x) => x + 1,
            Err(x) => x,
        };

        let occupied = &occupied[..end];

        let tdelay = self.tdelay(link);

        let mut offset = earliest_offset;
        let mut wait_time = 0;
        let mut accdelay_curr = accdelay + self.pdelay(link) + self.ldelay(link) + tdelay;

        let n = (max_time / self.period) as usize;

        let mut buf = Vec::new();

        let slots = loop {
            buf.clear();
            buf.push((0, 0));
            buf.extend(
                (0..hyper_period / self.period)
                    .map(|i| i * self.period + offset)
                    .map(|start| (start, start + tdelay)),
            );

            let mut exceed = false;

            if let Some(last) = buf.last().cloned() {
                if last.1 > hyper_period {
                    exceed = true;
                    buf[0] = (0, last.1 - hyper_period);
                    *buf.last_mut().unwrap() = (last.0, hyper_period);
                }
            }

            let slots = match exceed {
                true =>  &buf[..n + 1],
                false => &buf[1..n + 1],
            };

            match required_increase(slots, occupied) {
                0 => {
                    break Some(match exceed {
                        true =>  &buf[..],
                        false => &buf[1..],
                    });
                }
                inc => {
                    wait_time += inc;
                    offset += inc;
                    accdelay_curr += inc;

                    if offset >= self.period {
                        offset -= self.period;
                    };

                    if wait_time >= self.period {
                        break None;
                    }

                    if accdelay_curr > self.max_latency {
                        break None;
                    }
                }
            }
        };

        match slots {
            Some(slots) => {
                self.link_accdelays().insert(link.id, accdelay_curr);

                let link_offsets = self.link_offsets();
                link_offsets.insert(link.id, offset);

                if link_offsets.len() == self.links.len() {
                    self.schedule_done.set(true);
                }

                let slot_end = earliest + wait_time + tdelay;

                if fifo {
                    self.arrival().insert(link.id.1, (slot_end + self.pdelay(link)) % self.period);
                    if !ensure_fifo(self, link) {
                        return None;
                    }
                }

                link.merge_slots(slots);

                Some(slot_end + self.ldelay(link))
            }
            _ => {
                self.schedule_fail.set(true);
                None
            }
        }
    }

    fn accdelay(&self, link: &Link) -> u64 {
        match self.predecessors.get(&link.id) {
            Some(pred) => pred.iter().map(|l| self.accdelay_of(l)).max().unwrap_or(0),
            _ => 0,
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
        let mut outdegree: FxHashMap<LinkID, u32> = new!();
        for &(pred, _succ) in &self.routes {
            *outdegree.entry(pred).or_insert(0) += 1;
        }

        let mut result = FxHashMap::default();
        let mut queue = VecDeque::new();

        for id in self.links.keys() {
            if !outdegree.contains_key(id) {
                let link = &links[id];
                let delay = self.pdelay(link) + self.tdelay(link) + self.ldelay(link);
                result.insert(*id, delay);
                queue.push_back(*id);
            }
        }

        while let Some(curr) = queue.pop_front() {
            if let Some(preds) = self.predecessors.get(&curr) {
                for &pred in preds {
                    if let Some(count) = outdegree.get_mut(&pred) {
                        *count -= 1;
                    }

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
                    
                    if outdegree.get(&pred).map_or(true, |&count| count == 0) {
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

fn check_fifo(a: & Flow, b: & Flow, link: &Link) -> bool {
    match (a.arrival().get(&link.id.0), b.arrival().get(&link.id.0)) {
        (Some(&a_arrival_offset), Some(&b_arrival_offset)) => {
            let a_departure_offset = a.offset_of(&link.id);
            let b_departure_offset = b.offset_of(&link.id);

            let hyper_period = lcm(a.period, b.period);

            let generate_instances = |period: u64, arr_offset: u64, dep_offset: u64| -> Vec<(u64, u64)> {
                let rel_dep = if dep_offset < arr_offset {
                    dep_offset + period
                } else {
                    dep_offset
                };

                let count = hyper_period / period; 

                let mut instances = Vec::with_capacity(count as usize + 1);

                for i in 0..=count {
                    let base_time = i * period;
                    instances.push((base_time + arr_offset, base_time + rel_dep));
                }
                instances
            };

            let a_instances = generate_instances(a.period, a_arrival_offset, a_departure_offset);
            let b_instances = generate_instances(b.period, b_arrival_offset, b_departure_offset);

            let mut i = 0;
            let mut j = 0;
            let mut max_departure = 0;

            while i < a_instances.len() || j < b_instances.len() {
                let use_a = if i < a_instances.len() && j < b_instances.len() {
                    let (arr_a, dep_a) = a_instances[i];
                    let (arr_b, dep_b) = b_instances[j];
                    if arr_a < arr_b {
                        true
                    } else if arr_a > arr_b {
                        false
                    } else {
                        dep_a <= dep_b
                    }
                } else if i < a_instances.len() {
                    true
                } else {
                    false
                };

                let current_departure;
                
                if use_a {
                    current_departure = a_instances[i].1;
                    i += 1;
                } else {
                    current_departure = b_instances[j].1;
                    j += 1;
                }

                if current_departure < max_departure {
                    return false;
                }

                max_departure = current_departure;
            }

            true
        }
        _ => true,
    }
}

fn check_link_pcp(f: &Flow, link: &Link, pcp: i8) -> bool {
    let ref flows = processed_input().flows;
    for scheduled in link
        .flows
        .iter()
        .map(|f| &flows[f])
        .filter(|f| f.scheduled_link(link))
        .filter(|f | f.pcp.get() == pcp) {

        if std::ptr::eq(f, scheduled) {
            continue;
        }
        
        if !check_fifo(f, scheduled, link) {
            return false;
        }
    }
    true
}

fn check_flow_pcp(f: &Flow, pcp: i8) -> bool {
    let ref links = processed_input().links;
    for link in f.link_offsets().keys().map(|l| &links[l]) {
        if !check_link_pcp(f, link, pcp) {
            return false;
        }
    }
    true
}

fn ensure_fifo(f: &Flow, link: &Link) -> bool {
    if !check_link_pcp(f, link, f.pcp.get()) {
        for pcp in (0..8).rev() {
            if pcp != f.pcp.get() && check_flow_pcp(f, pcp) {
                f.pcp.set(pcp);
                return true;
            }
        }
        f.schedule_fail.set(true);
        f.pcp.set(-1);
        return false;
    }

    true
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
    pub fn breakloop(&mut self, flow: &Flow, route_id: &FxHashMap<(FlowID, RouteID), EdgeIndex>) {
        flow.breakloop.set(true);

        for route in flow.routes.iter() {
            if let Some(edge) = route_id.get(&(flow.id, *route)) {
                self.graph.remove_edge(*edge);
            }
        }

        self.remove_isolated_nodes();
    }

    pub fn remove_isolated_nodes(&mut self) {
        self.graph.retain_nodes(|graph, node| {
            let in_degree = graph.neighbors_directed(node, petgraph::Incoming).count();
            let out_degree = graph.neighbors_directed(node, petgraph::Outgoing).count();
            in_degree > 0 || out_degree > 0
        });
    }
}

fn required_increase(slots: &[(u64, u64)], occupied: &[(u64, u64)]) -> u64 {
    if slots.is_empty() || occupied.is_empty() {
        return 0;
    }

    let mut i = 0;
    let mut j = 0;
    let mut max_inc: u64 = 0;

    while i < slots.len() && j < occupied.len() {
        let (s1, e1) = slots[i];
        let (s2, e2) = occupied[j];

        if e1 <= s2 {
            i += 1;
        } else if e2 <= s1 {
            let next_j = j + 1;
            if next_j >= occupied.len() {
                break;
            }

            let search_slice = &occupied[next_j..];
            j = next_j
                + match search_slice.binary_search_by_key(&s1, |&(_, occ_e)| occ_e) {
                    Ok(pos) => pos + 1,
                    Err(pos) => pos,
                };
        } else {
            let inc = e2 - s1;

            if inc > max_inc {
                max_inc = inc;
            }

            if e1 < e2 {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    max_inc
}

pub fn sort_hops(hops: &FxHashSet<LinkID>) -> (Vec<LinkID>, FxHashMap<LinkID, Vec<LinkID>>) {
    let mut adjacency_list: UstrMap<Vec<Ustr>> = new!();
    let mut in_degree: UstrMap<usize> = new!();
    let mut predecessors: FxHashMap<LinkID, Vec<LinkID>> = new!();

    for &(from, to) in hops {
        adjacency_list.entry(from).or_default().push(to);
        *in_degree.entry(to).or_insert(0) += 1;
        in_degree.entry(from).or_insert(0);
    }

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

        sorted_hops.extend(hops.iter().filter(|(from, _)| *from == node));
    }

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
