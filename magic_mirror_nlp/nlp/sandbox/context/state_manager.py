class ContextManager:
    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_interaction(self, text):
        """Adds a new transcript to the context history."""
        self.history.append(text)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_string(self):
        """Returns the conversation history as a single formatted string."""
        if not self.history:
            return "No previous context. This is the first statement."
        
        context_str = "PREVIOUS STATEMENTS (For Context & Coreference):\n"
        for i, statement in enumerate(self.history):
            context_str += f"- Statement {i+1}: {statement}\n"
            
        return context_str

    def reset(self):
        """Clears the context history."""
        self.history = []
