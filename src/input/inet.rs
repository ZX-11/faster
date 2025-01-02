use fxhash::{FxHashMap, FxHashSet};
use lazy_static::lazy_static;
use serde::Deserialize;
use ustr::{ustr, Ustr, UstrMap};

use crate::model::{self, init_processed_input, ProcessedInput};

lazy_static! {
    pub static ref TIME_SCALE: UstrMap<u64> = UstrMap::from_iter([
        (ustr("ns"), 1),
        (ustr("us"), 1000),
        (ustr("ms"), 1000_000),
        (ustr("s"), 1000_000_000),
    ]);
}

#[derive(Deserialize, Debug)]
pub struct Device {
    pub name: Ustr,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Hop {
    pub current_node_name: Ustr,
    pub next_node_name: Ustr,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum Hops {
    SingleLevel(Vec<Hop>),
    MultiLevel(Vec<Vec<Hop>>),
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub struct Flow {
    pub end_devices: Vec<Ustr>,
    pub fixed_priority: Ustr,
    pub hard_constraint_time: u64,
    pub hard_constraint_time_unit: Ustr,
    pub hops: Hops,
    pub name: Ustr,
    pub packet_periodicity: u64,
    pub packet_periodicity_unit: Ustr,
    pub packet_size: u32,
    pub packet_size_unit: Ustr,
    pub priority_value: u32,
    pub source_device: Ustr,
    #[serde(rename = "type")]
    pub flow_type: Ustr,
    #[serde(skip)]
    pub hop_set: FxHashSet<(Ustr, Ustr)>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub struct Port {
    pub connects_to: Ustr,
    pub cycle_start: u32,
    pub cycle_start_unit: Ustr,
    pub maximum_slot_duration: u64,
    pub maximum_slot_duration_unit: Ustr,
    pub name: Ustr,
    pub port_speed: u32,
    pub port_speed_size_unit: Ustr,
    pub port_speed_time_unit: Ustr,
    pub schedule_type: Ustr,
    pub time_to_travel: f64,
    pub time_to_travel_unit: Ustr,
}

#[derive(Deserialize, Debug)]
pub struct Switch {
    pub name: Ustr,
    pub ports: Vec<Port>,
}

#[derive(Deserialize, Debug)]
pub struct Config {
    pub devices: Vec<Device>,
    pub flows: Vec<Flow>,
    pub switches: Vec<Switch>,
}

type Sequence = UstrMap<u32>;

fn get_sequence(seq: &Sequence, id: Ustr) -> u32 {
    match seq.get(&id) {
        Some(&s) => s,
        _ => {
            for (pat, s) in seq {
                if regex::Regex::new(pat.as_str())
                    .unwrap()
                    .is_match(id.as_str())
                {
                    return *s;
                }
            }
            0
        }
    }
}

pub fn process((filename, sequence): (&str, &str)) -> &'static ProcessedInput {
    let mut input: Config = json5::from_str(&std::fs::read_to_string(filename).unwrap()).unwrap();
    let seq: Sequence = match sequence {
        "" => Default::default(),
        _ => serde_json::from_slice(&std::fs::read(sequence).unwrap()).unwrap(),
    };
    let p = init_processed_input();

    p.devices = input
        .devices
        .iter()
        .map(|d| {
            (
                d.name,
                model::Device {
                    pdelay: 0,
                    end_device: true,
                },
            )
        })
        .chain(input.switches.iter().map(|sw| {
            (
                sw.name,
                model::Device {
                    pdelay: 0,
                    end_device: false,
                },
            )
        }))
        .collect();

    for flow in input.flows.iter_mut() {
        flow.hop_set = match &flow.hops {
            Hops::SingleLevel(hops) => hops
                .iter()
                .map(|h| (h.current_node_name, h.next_node_name))
                .collect(),
            Hops::MultiLevel(hops) => hops
                .iter()
                .flat_map(|level| {
                    level
                        .iter()
                        .map(|h| (h.current_node_name, h.next_node_name))
                })
                .collect(),
        };
    }

    p.links = input
        .switches
        .iter()
        .flat_map(|sw| {
            sw.ports.iter().map(|port| {
                (
                    (sw.name, port.connects_to),
                    model::Link {
                        id: (sw.name, port.connects_to),
                        from: p.devices[&sw.name],
                        _to: p.devices[&port.connects_to],
                        delay: (port.time_to_travel * TIME_SCALE[&port.time_to_travel_unit] as f64)
                            as u32,
                        speed: port.port_speed
                            * if port.port_speed_size_unit == "bit" {
                                1
                            } else {
                                8
                            },
                        flows: input
                            .flows
                            .iter()
                            .filter(|f| f.hop_set.contains(&(sw.name, port.connects_to)))
                            .map(|f| f.name)
                            .collect(),
                        ..Default::default()
                    },
                )
            })
        })
        .chain(input.switches.iter().flat_map(|sw| {
            sw.ports
                .iter()
                .filter(|port| p.devices[&port.connects_to].end_device)
                .map(|port| {
                    (
                        (port.connects_to, sw.name),
                        model::Link {
                            id: (port.connects_to, sw.name),
                            from: p.devices[&port.connects_to],
                            _to: p.devices[&sw.name],
                            delay: (port.time_to_travel
                                * TIME_SCALE[&port.time_to_travel_unit] as f64)
                                as u32,
                            speed: port.port_speed
                                * if port.port_speed_size_unit == "bit" {
                                    1
                                } else {
                                    8
                                },
                            flows: input
                                .flows
                                .iter()
                                .filter(|f| f.hop_set.contains(&(port.connects_to, sw.name)))
                                .map(|f| f.name)
                                .collect(),
                            ..Default::default()
                        },
                    )
                })
        }))
        .collect();

    p.flows = input
        .flows
        .iter()
        .map(|f| match &f.hops {
            Hops::SingleLevel(_) => {
                let (sorted_hops, predecessors) = model::sort_hops(&f.hop_set);
                (
                    f.name,
                    model::Flow {
                        id: f.name,
                        length: f.packet_size / if f.packet_size_unit == "bit" { 8 } else { 1 },
                        period: f.packet_periodicity * TIME_SCALE[&f.packet_periodicity_unit],
                        max_latency: f.hard_constraint_time
                            * TIME_SCALE[&f.hard_constraint_time_unit],
                        sequence: get_sequence(&seq, f.name),
                        links: sorted_hops.iter().map(|h| (*h, &p.links[h])).collect(),
                        predecessors,
                        ..Default::default()
                    }
                    .generate_remain_min_delay(&p.links)
                    .generate_routes(),
                )
            }
            Hops::MultiLevel(hops) => (
                f.name,
                model::Flow {
                    id: f.name,
                    length: f.packet_size / if f.packet_size_unit == "bit" { 8 } else { 1 },
                    period: f.packet_periodicity * TIME_SCALE[&f.packet_periodicity_unit],
                    max_latency: f.hard_constraint_time * TIME_SCALE[&f.hard_constraint_time_unit],
                    sequence: get_sequence(&seq, f.name),
                    links: hops
                        .iter()
                        .flat_map(|level| {
                            level.iter().map(|h| {
                                (
                                    (h.current_node_name, h.next_node_name),
                                    &p.links[&(h.current_node_name, h.next_node_name)],
                                )
                            })
                        })
                        .collect(),
                    predecessors: {
                        let mut result: FxHashMap<model::LinkID, Vec<model::LinkID>> =
                            FxHashMap::default();
                        for level in hops.iter() {
                            for hop in level.windows(2) {
                                result
                                    .entry((hop[1].current_node_name, hop[1].next_node_name))
                                    .or_default()
                                    .push((hop[0].current_node_name, hop[0].next_node_name));
                            }
                        }
                        result
                    },
                    routes: {
                        let mut result: Vec<model::RouteID> = Vec::new();
                        for level in hops.iter() {
                            for hop in level.windows(2) {
                                assert_eq!(hop[0].next_node_name, hop[1].current_node_name);
                                result.push((
                                    (hop[0].current_node_name, hop[0].next_node_name),
                                    (hop[1].current_node_name, hop[1].next_node_name),
                                ));
                            }
                        }
                        result
                    },
                    ..Default::default()
                }
                .generate_remain_min_delay(&p.links),
            ),
        })
        .collect();

    p.addition = Some(Box::new(input));

    return p;
}
