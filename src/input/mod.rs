pub mod fast;
pub mod inet;
pub mod json;

const HELP: &str = r#"Usage: faster [OPTIONS]

Options:
  --input-inet <file> [sequence]           Specify the input file for the INET input type and sequence.
  --input-json <file>                      Specify the input file for the JSON input type.
  --input-fast <device> <flow> <flowlink>  Specify device, flow, and flowlink files for the FAST input type.
  --output-inet <file>                     Specify the output file for the INET output type. Must be used after --input-inet.
  --output-console                         Output to the console.
  --help                                   Display this help message."#;

#[macro_export]
macro_rules! process_input {
    ($input_type:expr => { $($variant:ident),* }) => {
        match $input_type {
            $(InputType::$variant(args) => {
                $crate::input::$variant::process(args)
            })*
        }
    };
}

#[allow(unused_parens)]
#[allow(non_camel_case_types)]
pub enum InputType<'a> {
    inet((&'a str, &'a str)),
    json(&'a str),
    fast((&'a str, &'a str, &'a str)),
}

#[allow(unused_parens)]
pub enum OutputType<'a> {
    Inet((&'a str)),
    Console,
}

#[allow(unused_parens)]
pub fn parse_args<'a>(
    mut args: impl Iterator<Item = &'a str>,
) -> Result<(InputType<'a>, OutputType<'a>), Box<dyn std::error::Error>> {
    let mut input_type: Option<InputType<'a>> = None;
    let mut output_type: Option<OutputType<'a>> = None;
    while let Some(arg) = args.next() {
        match arg {
            "--input-inet" => {
                let filename = args.next().ok_or("Missing input file for --input-inet")?;
                let sequence = args.next().unwrap_or("");
                input_type = Some(InputType::inet((filename, sequence)));
            }
            "--input-json" => {
                let filename = args.next().ok_or("Missing input file for --input-json")?;
                input_type = Some(InputType::json(filename));
            }
            "--input-fast" => {
                let device = args.next().ok_or("Missing device file for --input-fast")?;
                let flow = args.next().ok_or("Missing flow file for --input-fast")?;
                let flowlink = args
                    .next()
                    .ok_or("Missing flowlink file for --input-fast")?;
                input_type = Some(InputType::fast((device, flow, flowlink)));
            }
            "--output-inet" => {
                let filename = args.next().ok_or("Missing output file for --output-inet")?;
                if let Some(InputType::inet(_)) = input_type {
                    output_type = Some(OutputType::Inet((filename)));
                } else {
                    return Err("--input-inet must be specified before --output-inet".into());
                }
            }
            "--output-console" => {
                output_type = Some(OutputType::Console);
            }
            "-h" | "--help" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {}", arg).into()),
        }
    }

    match (input_type, output_type) {
        (Some(input), Some(output)) => Ok((input, output)),
        _ => Err("Missing input or output type".into()),
    }
}
