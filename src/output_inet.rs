use std::{fs::File, io::Write};

use fxhash::FxHashMap;
use regex::Regex;
use serde::Serialize;
use smallvec::SmallVec;
use ustr::{Ustr, UstrMap};

use crate::{
    input_inet::{self, time_scale},
    model::ProcessedInput,
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

pub fn output(processed_input: &ProcessedInput, filename: &str) {
    // 重新获取原始配置
    let input = processed_input
        .addition
        .as_ref()
        .unwrap()
        .downcast_ref::<input_inet::Config>()
        .unwrap();

    let mut output = Output {
        flows: Vec::with_capacity(input.flows.len()),
        switches: Vec::with_capacity(input.switches.len()),
    };

    // 处理流信息
    for original_flow in &input.flows {
        let model_flow = &processed_input.flows[&original_flow.name];
        let schedule = model_flow.schedule();
        let schedule = schedule.borrow();
        let first_sending_time = schedule.link_offsets.values().min().cloned().unwrap();
        let (last_link, last_offset) = schedule
            .link_offsets
            .iter()
            .max_by_key(|(_, offset)| *offset)
            .unwrap();

        output.flows.push(Flow {
            name: original_flow.name,
            first_sending_time: first_sending_time as f64 / 1_000.0,
            average_latency: (last_offset + model_flow.tdelay(model_flow.link_map[last_link])
                - first_sending_time) as f64
                / 1_000.0,
            jitter: 0.0,
            flow_priority: original_flow.priority_value,
            flow_periodicity: (original_flow.packet_periodicity
                * time_scale(original_flow.packet_periodicity_unit)
                / 1_000) as u32,
        });
    }

    // flow_offset[flow_name][from][port] = offset
    let mut flow_offset: UstrMap<UstrMap<FxHashMap<u32, u64>>> = UstrMap::default();
    
    // 处理交换机和端口信息
    for switch in &input.switches {
        let mut ports = Vec::with_capacity(switch.ports.len());

        for port in &switch.ports {
            let mut priority_slots_data = Vec::with_capacity(8);

            // 初始化优先级时隙数据
            let mut priority_slots: FxHashMap<u32, Vec<(u64,u64)>> = FxHashMap::default();

            let link_id = (switch.name, port.connects_to);

            // 遍历流并收集端口调度信息
            for flow in input.flows.iter().filter(|f| f.hop_set.contains(&link_id)) {
                // 获取流的调度信息
                let model_flow = &processed_input.flows[&flow.name];
                if let Some(offset) = model_flow.schedule().borrow().link_offsets.get(&link_id) {
                    priority_slots
                        .entry(flow.priority_value as u32)
                        .or_default()
                        .push((*offset, model_flow.tdelay(model_flow.link_map[&link_id])));
                    flow_offset
                        .entry(flow.name)
                        .or_default()
                        .entry(switch.name)
                        .or_default()
                        .insert(eth_index(port.name.into()).unwrap(), *offset);
                }
            }

            // 转换优先级时隙数据
            for (&priority, slots) in &priority_slots {
                priority_slots_data.push(PrioritySlotsData {
                    slots_data: merge_slots(&mut slots.clone()).into_iter().map(|(start, duration)| SlotsData {
                        slot_start: start as f64 / 1_000.0,
                        slot_duration: duration as f64 / 1_000.0,
                    }).collect(),
                    priority,
                });
            }

            let mut occupied = priority_slots
                .values()
                .flat_map(|v| v.iter())
                .cloned()
                .collect::<SmallVec<[_; 1024]>>();
            
            let occupied= merge_slots(&mut occupied);

            let cycle_duration = port.maximum_slot_duration
                * time_scale(port.maximum_slot_duration_unit);

            let mut free = Vec::with_capacity(occupied.len()+1);

            if occupied.len() > 0 {
                if occupied[0].0 != 0 {
                    free.push(SlotsData {
                        slot_start: 0.0,
                        slot_duration: occupied[0].0 as f64 / 1_000.0,
                    });
                }
                for i in 0..occupied.len() {
                    let start = occupied[i].0 + occupied[i].1;
                    let end = if i == occupied.len() - 1 {
                        cycle_duration
                    } else {
                        occupied[i + 1].0
                    };
                    free.push(SlotsData {
                        slot_start: start as f64 / 1_000.0,
                        slot_duration: (end - start) as f64 / 1_000.0,
                    });
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

    let flow_period: UstrMap<u64> = processed_input.flows.iter().map(|(name, flow)| (*name, flow.period)).collect();

    output_file(serde_json::to_string_pretty, &output, filename).unwrap();
    output_file(serde_json::to_string_pretty, &flow_offset, "flow-offset.json").unwrap();
    output_file(serde_json::to_string_pretty, &flow_period, "flow-period.json").unwrap();
}


fn output_file<S, T, E>(ser: S, value: &T, filename: &str) -> Result<usize, Box<dyn std::error::Error>>
where
    T: ?Sized + Serialize,
    S: Fn(&T) -> Result<String, E>,
    E: 'static + std::error::Error + Send + Sync,
{
    Ok(File::create(filename)?.write(ser(value)?.as_bytes())?)
}

fn merge_slots(slots_data: &mut [(u64,u64)]) -> Vec<(u64,u64)> {
    if slots_data.len() <= 1 {
        return slots_data.to_vec();
    }
    slots_data.sort_by_key(|&(start,_)| start);
    let mut merged_slots = Vec::with_capacity(slots_data.len());
    merged_slots.push(slots_data[0]);

    for (start,duration) in slots_data.into_iter().skip(1) {
        let last = merged_slots.last_mut().unwrap();
        if last.0 + last.1 == *start {
            last.1 += *duration;
        } else {
            merged_slots.push((*start, *duration));
        }
    }
    merged_slots
}

fn eth_index(input: &str) -> Option<u32> {
    let re = Regex::new(r"eth\[(\d+)\]").unwrap();
    re.captures(input)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<u32>().ok())
}