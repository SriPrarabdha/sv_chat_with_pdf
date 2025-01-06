from main import conversational_rag_chain
import litserve as ls

class RagAgentAPI(ls.LitAPI):
    def setup(self, api_key:str, flash_rank_model:str, embeddings:str):
    
        self.flash_rank_model = flash_rank_model
        self.api_key = api_key
        self.embedding = embeddings

    def decode_request(self, request):
        # Extract prompt from request
        prompt = request["prompt"]
        return prompt

    def predict(self, prompt):
        
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )

        return response

    def encode_response(self, response):
        answer = response["answer"]
        chat_history = response["chat_history"]

        context = {}
        for i,document in enumerate(response['context']):
            context[i] = document.page_content
        
        return {"answer": answer, "chat_history":chat_history, "context": [context]}


if __name__ == "__main__":
    
    api = RagAgentAPI()
    server = ls.LitServer(api)
    server.run(port=8000)