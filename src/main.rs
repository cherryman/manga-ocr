use base64::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub logprobs: Option<LogProbs>,
    #[serde(rename = "finish_reason")]
    pub finish_reason: FinishReason,
}

#[derive(Deserialize)]
pub struct LogProbs {
    pub content: Option<Vec<TokenInfo>>,
}

#[derive(Deserialize)]
pub struct TokenInfo {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<i32>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Deserialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<i32>>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    #[serde(rename = "function_call")]
    FunctionCall,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub enum Msg {
    User { content: String },
}

#[derive(clap::Parser)]
pub struct Cli {
    #[arg()]
    images: Vec<std::path::PathBuf>,
}

fn main() -> eyre::Result<()> {
    // https://platform.openai.com/docs/api-reference/chat/create

    // TODO: detect file type and set in url.
    // TODO: error handling for certain things.

    let Ok(api_token) = std::env::var("OPENAI_API_KEY") else {
        eyre::bail!("OPENAI_API_KEY environment variable not set")
    };

    let Cli { images } = <Cli as clap::Parser>::parse();

    for xs in images.chunks(8) {
        let xs = xs.iter().map(std::fs::read).collect::<Result<Vec<_>, _>>()?;
        let chat = fetch(&api_token, &xs)?;
        for choice in chat.choices {
            if choice.finish_reason != FinishReason::Stop {
                continue;
            }

            for m in choice.message.content.split('\n') {
                println!("{}", m);
            }
        }
    }

    Ok(())
}

fn fetch(api_token: &str, images: &[impl AsRef<[u8]>]) -> eyre::Result<ChatCompletion> {
    let mut content = Vec::with_capacity(images.len() + 1);

    content.push(serde_json::json! {
        {
            "type": "text",
            "text": "List all Japanese sentences in these images, separated by a newline. Say nothing else.",
        }
    });

    for x in images {
        let url = BASE64_STANDARD.encode(x);
        let url = format!("data:image/jpeg;base64,{}", url);
        content.push(serde_json::json! {
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": "high",
                },
            }
        });
    }

    ureq::post("https://api.openai.com/v1/chat/completions")
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_token))
        .send_json(serde_json::json! {
            {
                "model": "gpt-4o",
                "max_tokens": 600,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            }
        })?
        .into_json()
        .map_err(Into::into)
}
