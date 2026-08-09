import chromadb

import gradio as gr
from chromadb.utils import embedding_functions


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.Client()
collection_name = "my_grocery_collection"


collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for storing grocery data"},
    configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef},
)

texts = [
    "fresh red apples",
    "organic bananas",
    "ripe mangoes",
    "whole wheat bread",
    "farm-fresh eggs",
    "natural yogurt",
    "frozen vegetables",
    "grass-fed beef",
    "free-range chicken",
    "fresh salmon fillet",
    "aromatic coffee beans",
    "pure honey",
    "golden apple",
    "red fruit",
]

ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]


if collection.count() == 0:
  collection.add(
      documents=texts,
      metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
      ids=ids,
  )



def search_grocery(query_term, top_k):
  if not query_term.strip():
    return "Please enter a search query text."

  try:
    results = collection.query(
        query_texts=[query_term], n_results=int(top_k)
    )

    if not results or not results["ids"] or len(results["ids"][0]) == 0:
      return f'No documents found similar to "{query_term}"'

    output = f"### Top {top_k} Search Results for '{query_term}':\n\n"

    for i in range(min(int(top_k), len(results["ids"][0]))):
      doc_id = results["ids"][0][i]
      score = results["distances"][0][i]
      text = results["documents"][0][i]

      output += f"**{i+1}. Document ID:** `{doc_id}`  \n"
      output += f"**Item:** {text}  \n"
      output += f"**Cosine Distance Score:** `{score:.4f}`  \n"
      output += "---\n"

    return output

  except Exception as error:
    return f"Error during search: {error}"



with gr.Blocks(title="Grocery Similarity Search") as demo:
  gr.Markdown("# 🛒 Grocery Vector Similarity Search Dashboard")
  gr.Markdown(
      "Enter a query term to find the closest items in the Chroma Vector Database."
  )

  with gr.Row():
    with gr.Column():
      query_input = gr.Textbox(
          label="Search Query",
          placeholder="e.g., sweet red fruit, breakfast item, meats...",
      )
      top_k_slider = gr.Slider(
          minimum=1, maximum=5, value=3, step=1, label="Select Top K Results"
      )
      search_button = gr.Button("🔍 Search Database", variant="primary")

    with gr.Column():
      output_display = gr.Markdown(
          label="Search Results", value="Search results will appear here..."
      )


  search_button.click(
      fn=search_grocery,
      inputs=[query_input, top_k_slider],
      outputs=output_display,
  )


if __name__ == "__main__":
  demo.launch(share=True)