use std::fs;

use fxhash::FxHashSet;
use serde::Deserialize;
use ustr::Ustr;

use crate::model::{self, init_processed_input, ProcessedInput};

#[derive(Deserialize, Debug)]
pub struct Input {
    pub devices: Vec<Device>,
    pub flows: Vec<Flow>,
}

#[derive(Deserialize, Debug)]
pub struct Flow {
    pub name: Ustr,
 
    // require either route or hops
    pub route: Option<Vec<Ustr>>,
    pub hops: Option<FxHashSet<(Ustr, Ustr)>>,

    pub period: u64,
    pub length: u32,
    pub max_latency: u64,
    #[serde(default)]
    pub sequence: u32,
}

#[derive(Deserialize, Debug)]
pub struct Device {
    pub name: Ustr,
    pub pdelay: u64,
    pub links: Vec<Link>,
    #[serde(rename = "type")]
    pub device_type: Ustr, // "end" or "switch"
}

#[derive(Deserialize, Debug)]
pub struct Link {
    pub to: Ustr,
    pub ldelay: u32,
    pub speed: u32,
}

pub fn process(filename: &str) -> &'static ProcessedInput {
    let mut input: Input = serde_json::from_slice(fs::read(filename).unwrap().as_slice()).unwrap();
    let p = init_processed_input();

    // 如果使用不同的输入格式，可以重写以下的处理逻辑

    for f in input.flows.iter_mut() {
        if let Some(route) = &f.route {
            f.hops = Some(route.windows(2).map(|w| (w[0], w[1])).collect());
        }
    }

    p.devices = input
        .devices
        .iter()
        .map(|d| {
            (
                d.name,
                model::Device {
                    // id: d.name,
                    pdelay: d.pdelay,
                    end_device: d.device_type == "end",
                },
            )
        })
        .collect();

    p.links = input
        .devices
        .iter()
        .flat_map(|d| {
            d.links.iter().map(|ln| {
                (
                    (d.name, ln.to),
                    model::Link {
                        id: (d.name, ln.to),
                        from: p.devices[&d.name],
                        _to: p.devices[&ln.to],
                        delay: ln.ldelay,
                        speed: ln.speed,
                        flows: input
                            .flows
                            .iter()
                            .filter(|f| f.hops.as_ref().unwrap().contains(&(d.name, ln.to)))
                            .map(|f| f.name)
                            .collect(),
                    },
                )
            })
        })
        .collect();

    p.flows = input
        .flows
        .iter()
        .map(|f| {
            let (sorted_hops, predecessors) = model::sort_hops(f.hops.as_ref().unwrap());
            // 创建完整的流对象
            (
                f.name,
                model::Flow {
                    id: f.name,
                    length: f.length,
                    period: f.period,
                    max_latency: f.max_latency,
                    sequence: if f.sequence == 0 {
                        u32::MAX
                    } else {
                        f.sequence
                    },
                    links: sorted_hops.iter().map(|h| &p.links[h]).collect(),
                    link_map: sorted_hops.iter().map(|h| (*h, &p.links[h])).collect(),
                    predecessors,
                    ..Default::default()
                }
                .generate_remain_min_delay(&p.links)
                .generate_routes(),
            )
        })
        .collect();

    return p;
}
