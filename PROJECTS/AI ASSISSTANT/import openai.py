import openai
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os

class PersonalAIAssistant:
    def __init__(self):
        # Initialize OpenAI API (you'll need your own API key)
        openai.api_key = 
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
    def listen(self):
        """Listen to user's voice input"""
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except:
                return "Sorry, I couldn't understand that."
    
    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_ai_response(self, prompt):
        """Get response from OpenAI API"""
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def handle_command(self, command):
        """Handle different types of commands"""
        command = command.lower()
        
        if 'time' in command:
            current_time = datetime.datetime.now().strftime('%H:%M')
            return f"The current time is {current_time}"
            
        elif 'search' in command:
            search_term = command.replace('search', '').strip()
            webbrowser.open(f'https://www.google.com/search?q={search_term}')
            return f"Searching for {search_term}"
            
        else:
            # Get AI response for general queries
            return self.get_ai_response(command)
    
    def run(self):
        """Main loop for the assistant"""
        self.speak("Hello! I'm your personal AI assistant. How can I help you?")
        
        while True:
            command = self.listen()
            if command.lower() == 'exit':
                self.speak("Goodbye!")
                break
                
            response = self.handle_command(command)
            self.speak(response)

# Create and run the assistant
if __name__ == "__main__":
    assistant = PersonalAIAssistant()
    assistant.run()
