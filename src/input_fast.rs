use smallvec::SmallVec;
use ustr::{ustr, Ustr, UstrMap};

use crate::model::*;

pub fn process(
    device_path: &str,
    flow_path: &str,
    flowlink_path: &str,
) -> &'static ProcessedInput {
    let p = init_processed_input();
    let mut flow_hops: UstrMap<Vec<(Ustr, Ustr)>> = UstrMap::default();
    {
        // Process device.txt
        let device_file = std::fs::read_to_string(device_path).unwrap();
        let mut lines = device_file.lines();

        // Network parameters
        let network_params: SmallVec<[&str; 8]> =
            lines.next().unwrap().split_whitespace().collect();

        let speed = network_params[5].parse().unwrap();

        for line in lines {
            let parts: SmallVec<[&str; 6]> = line.split_whitespace().collect();

            match parts.as_slice() {
                &[id, _rdelay, _fdelay, pdelay, _sfdelay, end] => {
                    p.devices.insert(
                        id.into(),
                        Device {
                            pdelay: pdelay.parse().unwrap(),
                            end_device: end == "1",
                        },
                    );
                }
                &[_id, from, to, ldelay] => {
                    let from = ustr(from);
                    let to = ustr(to);

                    p.links.insert(
                        (from, to),
                        Link {
                            id: (from, to),
                            from: p.devices[&from],
                            _to: p.devices[&to],
                            delay: ldelay.parse().unwrap(),
                            speed,
                            flows: Vec::new(),
                        },
                    );
                }
                _ => (),
            }
        }
    }
    {
        // Process flowlink.txt
        let flowlink_file = std::fs::read_to_string(flowlink_path).unwrap();

        flow_hops.extend(flowlink_file.lines().enumerate().map(|(id, line)| {
            let nodes: SmallVec<[&str; 32]> = line.split_whitespace().collect();
            (
                (id + 1).to_string().into(),
                nodes
                    .windows(2)
                    .step_by(2)
                    .map(|w| (ustr(w[0]), ustr(w[1])))
                    .collect(),
            )
        }));

        for (id, hops) in &flow_hops {
            for h in hops {
                p.links.get_mut(h).unwrap().flows.push(*id);
            }
        }
    }
    {
        // Process flow.txt
        let flow_file = std::fs::read_to_string(flow_path).unwrap();

        for line in flow_file.lines().skip(1) {
            let parts: SmallVec<[&str; 8]> = line.split_whitespace().collect();

            match parts.as_slice() {
                &[id, length, period, max_latency, ..] => {
                    let id = ustr(id);
                    let hops = &flow_hops[&id];

                    p.flows.insert(
                        id,
                        Flow {
                            id,
                            length: length.parse().unwrap(),
                            period: period.parse().unwrap(),
                            max_latency: max_latency.parse().unwrap(),
                            sequence: u32::MAX,
                            links: hops.iter().map(|h| (*h, &p.links[h])).collect(),
                            predecessors: hops.windows(2).map(|w| (w[1], vec![w[0]])).collect(),
                            routes: hops.windows(2).map(|w| (w[0], w[1])).collect(),
                            ..Default::default()
                        }
                        .generate_remain_min_delay(&p.links),
                    );
                }
                _ => (),
            }
        }
    }
    p
}
