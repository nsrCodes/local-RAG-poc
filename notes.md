### embedding quality

Was initially using `nomic-embed-text` which was the most popular but in my experience:

- Pro: It was very fast at generating embedding which was useful for a seemless experience
    - Took 4 mins to embed the transcript (all 10 seasons) of the TBBT
- Con: The similarity search on the generated index rarely found the relevant search. especially if the query is vague and not specifically mentioned in the data that has been embedded

shifted to `mxbai-embed-large` since people mentioned in forum that it has better quality of embeddings, but I feel there is still a SOTA model that is yet to come. shifting to this increased the time to embed upto 7 mins for the same transcript dataset mentioned above

The common approach is to apply other optimizations on the records received via similarity search:

- fine tuning the embedding model for specific usecase
- Applying ranking algorithms for the retreived data

### Prompt

Tried several approaches like:

1. Trying to ask the model how to think. Then try to hide those intermediate steps of “answer formatting” in an attempt to emulate reasoning but was hard to control
2. Had to iterate the prompt in different ways when trying to add conversation history. Could be a lot better.
3. Reformating prompts based on the prompt template available in the model description increases the adherence to the prompt a lot