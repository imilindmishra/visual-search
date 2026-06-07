def build_prompt(products: list[dict], intent: str) -> str:
    product_block = "\n".join([
        f"- {p['name']} | Category: {p['category']} | Price: ₹{p['price']} | {p['description']} | Similarity: {p['score']}"
        for p in products
    ])

    intent_map = {
        "recommend": "Recommend the most suitable product from the list for the user. Explain why briefly.",
        "compare":   "Compare the top 3 products across price, category, and quality. Use a simple structure.",
        "authentic": "Assess whether the product appears genuine based on available signals. Note any concerns.",
    }

    task = intent_map.get(intent, intent_map["recommend"])

    return f"""You are a helpful product advisor. Use ONLY the products listed below — do not invent any others.

Products retrieved:
{product_block}

Task: {task}

Be concise and helpful. Answer in 3–5 sentences."""