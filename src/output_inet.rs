use std::{fs::File, io::Write};

use fxhash::FxHashMap;
use regex::Regex;
use serde::Serialize;
use ustr::{Ustr, UstrMap};

use crate::{
    input::inet::{Config, TIME_SCALE},
    model::processed_input,
};

#[derive(Serialize, Debug, Copy, Clone)]
#[serde(rename_all = "camelCase")]
struct SlotsData {
    slot_start: f64,
    slot_duration: f64,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct PrioritySlotsData {
    slots_data: Vec<SlotsData>,
    priority: u32,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Flow {
    name: Ustr,
    first_sending_time: f64,
    average_latency: f64,
    jitter: f64,
    flow_priority: u32,
    flow_periodicity: u32,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Port {
    name: Ustr,
    cycle_duration: u32,
    first_cycle_start: f64,
    priority_slots_data: Vec<PrioritySlotsData>,
    connects_to: Ustr,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Switch {
    name: Ustr,
    ports: Vec<Port>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Output {
    flows: Vec<Flow>,
    switches: Vec<Switch>,
}

pub fn output(filename: &str) {
    let p = processed_input();

    let input = p
        .addition
        .as_ref()
        .unwrap()
        .downcast_ref::<Config>()
        .unwrap();

    let mut output = Output {
        flows: Vec::with_capacity(input.flows.len()),
        switches: Vec::with_capacity(input.switches.len()),
    };

    // 处理流信息
    for original_flow in &input.flows {
        let model_flow = &p.flows[&original_flow.name];
        
        let first_sending_time = model_flow
            .link_offsets()
            .values()
            .min()
            .cloned()
            .unwrap();

        let (&last_link, &last_offset) = model_flow
            .link_offsets()
            .iter()
            .max_by_key(|(_, offset)| *offset)
            .unwrap();

        output.flows.push(Flow {
            name: original_flow.name,
            first_sending_time: first_sending_time as f64 / 1_000.0,
            average_latency: (last_offset + model_flow.tdelay(model_flow.links[&last_link])
                - first_sending_time) as f64
                / 1_000.0,
            jitter: 0.0,
            flow_priority: original_flow.priority_value,
            flow_periodicity: (original_flow.packet_periodicity
                * TIME_SCALE[&original_flow.packet_periodicity_unit]
                / 1_000) as u32,
        });
    }

    // flow_offset[flow_name][from][port] = offset
    let mut flow_offset: UstrMap<UstrMap<FxHashMap<u32, u64>>> = UstrMap::default();
    let mut last_slot_end = 0;

    for (id, flow) in &p.flows {
        for ((from, _), offset) in flow.link_offsets() {
            if p.devices[&from].end_device {
                flow_offset
                .entry(*id)
                .or_default()
                .entry(*from)
                .or_default()
                .insert(0, *offset);
            }
        }
    }

    // 处理交换机和端口信息
    for switch in &input.switches {
        let mut ports = Vec::with_capacity(switch.ports.len());

        for port in &switch.ports {
            let mut priority_slots_data = Vec::with_capacity(8);

            // 初始化优先级时隙数据
            let mut priority_slots: FxHashMap<u32, Vec<(u64, u64)>> = FxHashMap::default();

            let link_id = (switch.name, port.connects_to);

            // 遍历流并收集端口调度信息
            for flow in input.flows.iter().filter(|f| f.hop_set.contains(&link_id)) {
                // 获取流的调度信息
                let model_flow = &p.flows[&flow.name];
                let offset = model_flow.offset_of(&link_id);
                priority_slots
                    .entry(flow.priority_value)
                    .or_default()
                    .push((offset, model_flow.tdelay(model_flow.links[&link_id])));
                flow_offset
                    .entry(flow.name)
                    .or_default()
                    .entry(switch.name)
                    .or_default()
                    .insert(eth_index(port.name.into()).unwrap(), offset);
            }

            // 转换优先级时隙数据
            for (&priority, slots) in &priority_slots {
                priority_slots_data.push(PrioritySlotsData {
                    slots_data: merge_slots(slots.clone())
                        .into_iter()
                        .inspect(|&(start, duration)| last_slot_end = last_slot_end.max(start + duration))
                        .map(|(start, duration)| SlotsData {
                            slot_start: start as f64 / 1_000.0,
                            slot_duration: duration as f64 / 1_000.0,
                        })
                        .collect(),
                    priority,
                });
            }

            let occupied = merge_slots(
                priority_slots
                    .values()
                    .flat_map(|v| v.iter())
                    .cloned()
                    .collect(),
            );

            let cycle_duration =
                port.maximum_slot_duration * TIME_SCALE[&port.maximum_slot_duration_unit];

            let mut free = Vec::with_capacity(occupied.len() + 1);

            let protect = 1000;

            if occupied.len() > 0 {
                if occupied[0].0 > protect {
                    free.push(SlotsData {
                        slot_start: 0.0,
                        slot_duration: (occupied[0].0 - protect) as f64 / 1_000.0,
                    });
                }
                for i in 0..occupied.len() {
                    let start = occupied[i].0 + occupied[i].1;
                    let end = if i == occupied.len() - 1 {
                        cycle_duration
                    } else {
                        occupied[i + 1].0
                    };
                    if end > start + protect {
                        free.push(SlotsData {
                            slot_start: start as f64 / 1_000.0,
                            slot_duration: (end - start - protect) as f64 / 1_000.0,
                        });
                    }
                }
            } else {
                free.push(SlotsData {
                    slot_start: 0.0,
                    slot_duration: cycle_duration as f64 / 1_000.0,
                });
            }

            // 添加默认优先级时隙
            priority_slots_data.push(PrioritySlotsData {
                slots_data: vec![SlotsData {
                    slot_start: 0.0,
                    slot_duration: cycle_duration as f64 / 1_000.0,
                }],
                priority: 2,
            });

            priority_slots_data.push(PrioritySlotsData {
                slots_data: free,
                priority: 0,
            });

            ports.push(Port {
                name: port.name,
                cycle_duration: cycle_duration as u32 / 1000,
                first_cycle_start: 0.0,
                priority_slots_data,
                connects_to: port.connects_to,
            });
        }

        output.switches.push(Switch {
            name: switch.name,
            ports,
        });
    }

    let flow_period: UstrMap<u64> = p.flows.iter().map(|(name, flow)| (*name, flow.period)).collect();

    output_file(serde_json::to_string_pretty, &output, filename).unwrap();
    output_file(serde_json::to_string_pretty, &flow_offset, "flow-offset.json").unwrap();
    output_file(serde_json::to_string_pretty, &flow_period, "flow-period.json").unwrap();
    println!("Cycle completion offset: {} ns", last_slot_end);
}

fn output_file<S, T, E>(
    ser: S,
    value: &T,
    filename: &str,
) -> Result<usize, Box<dyn std::error::Error>>
where
    T: ?Sized + Serialize,
    S: Fn(&T) -> Result<String, E>,
    E: 'static + std::error::Error + Send + Sync,
{
    Ok(File::create(filename)?.write(ser(value)?.as_bytes())?)
}

fn merge_slots(mut slots_data: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if slots_data.len() <= 1 {
        return slots_data;
    }
    slots_data.sort_unstable_by_key(|&(start, _)| start);

    let mut w = 0;

    for r in 1..slots_data.len() {
        unsafe {
            let curr = slots_data.get_unchecked(w);
            let next = slots_data.get_unchecked(r);

            if curr.0 + curr.1 == next.0 {
                slots_data.get_unchecked_mut(w).1 += next.1;
            } else {
                w += 1;
                if w != r {
                    *slots_data.get_unchecked_mut(w) = *next;
                }
            }
        }
    }

    // 返回合并后的数组长度
    slots_data.truncate(w + 1);
    slots_data
}

fn eth_index(input: &str) -> Option<u32> {
    Regex::new(r"eth\[(\d+)\]")
        .unwrap()
        .captures(input)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse().ok())
}
