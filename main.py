from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


def chunk_text(text, chunk_size):
    """
    Split the text into chunks of size `chunk_size`.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size
    return chunks


f = open("sre2.txt").read()
text = f

text_chunks = chunk_text(text, 1024)

torch.manual_seed(0)

generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")


def generate_summary_for_chunk(chunk):
    prompt = f"""You are an expert in Site Reliability Engineering (SRE) and an experienced software engineering professor. 
    Your task is to create a detailed and structured teaching transcript for a 30-minute lesson. 
    This transcript will be used to teach a course on SRE to students with basic knowledge of software engineering.

    Here is a sub-paragraph of a lecture transcript about Site Reliability Engineering (SRE): {chunk}


    Here is a structure template for you need to include in the transcript for Site Reliability Engineering (SRE) :
    1. **Introduction**:
        - Start with a brief history of Site Reliability Engineering.
        - Explain its importance in modern technology and the role of SRE in improving system reliability.

    2. **Key Concepts**:
        - Define critical concepts, including reliability, availability, and scalability.
        - Explain SLOs (Service Level Objectives), SLAs (Service Level Agreements), and error budgets, providing practical examples.

    3. **Main Practices of SRE**:
        - Discuss key practices such as monitoring, incident management, and postmortems.
        - Highlight the significance of automation in reducing toil and improving efficiency.

    4. **Team Dynamics**:
        - Explain how SRE teams collaborate with development and operations teams.
        - Discuss how SRE bridges gaps between these teams to foster reliability and innovation.

    5. **Case Studies and Real-World Applications**:
        - Provide examples of how companies like Google have successfully implemented SRE principles.
        - Discuss challenges faced in SRE and how they can be mitigated.

    6. **Conclusion**:
        - Recap the key takeaways of the lesson.
        - Emphasize the future of SRE and its evolving role in cloud-native environments.

    Ensure the transcript is detailed, logical, and coherent. 
    Write it in a conversational tone suitable for a classroom setting. 
    Use bullet points or numbered lists where appropriate to organize the content. 
    Keep the language simple and engaging for students."""

    summary = generator(
        prompt,
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        truncation=True,
        num_return_sequences=1,
        pad_token_id=50256
    )
    return (summary[0]["generated_text"])


all_summaries = []
for i, chunk in enumerate(text_chunks):
    print(f"Processing chunk {i + 1}/{len(text_chunks)}...")
    summary = generate_summary_for_chunk(chunk)
    all_summaries.append(summary)


for i, summary in enumerate(all_summaries):
    print(f"Summary for Chunk {i + 1}:\n{summary}\n")