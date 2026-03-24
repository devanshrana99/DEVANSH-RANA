"""
AIML-based Traffic Chatbot Engine
Processes natural language queries about traffic conditions
"""

import re
import os
import xml.etree.ElementTree as ET
from datetime import datetime


class AIMLEngine:
    """
    Lightweight AIML 1.0 interpreter for the Traffic Bot.
    Parses the traffic_bot.aiml file and matches user inputs to responses.
    """

    def __init__(self, aiml_file: str):
        self.patterns = []   # List of (compiled_regex, template_text)
        self.srais = {}      # SRAI redirects
        self._load_aiml(aiml_file)

    def _wildcard_to_regex(self, pattern: str) -> re.Pattern:
        """Convert AIML pattern (with *) to regex"""
        escaped = re.escape(pattern)
        escaped = escaped.replace(r'\*', '(.*)')
        return re.compile(f'^{escaped}$', re.IGNORECASE)

    def _resolve_template(self, template_el) -> str:
        """Extract text and handle <srai> tags"""
        parts = []
        if template_el.text:
            parts.append(template_el.text)
        for child in template_el:
            if child.tag == 'srai':
                srai_key = (child.text or '').strip().upper()
                parts.append(f'__SRAI__{srai_key}__SRAI__')
            if child.tail:
                parts.append(child.tail)
        return ''.join(parts)

    def _load_aiml(self, filepath: str):
        """Parse AIML XML and build pattern matcher"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            for category in root.findall('category'):
                pattern_el = category.find('pattern')
                template_el = category.find('template')

                if pattern_el is None or template_el is None:
                    continue

                pattern_text = (pattern_el.text or '').strip().upper()
                template_text = self._resolve_template(template_el)

                regex = self._wildcard_to_regex(pattern_text)
                self.patterns.append((regex, pattern_text, template_text))

        except Exception as e:
            print(f"AIML load error: {e}")
            # Fallback patterns
            self.patterns = []

    def _get_response(self, user_input: str, depth=0) -> str:
        """Match input against patterns and return response"""
        if depth > 5:  # Prevent infinite SRAI loops
            return "I'm here to help with traffic questions!"

        normalized = user_input.strip().upper()

        for regex, pattern_text, template_text in self.patterns:
            match = regex.match(normalized)
            if match:
                # Handle SRAI redirects
                if '__SRAI__' in template_text:
                    srai_match = re.search(r'__SRAI__(.+?)__SRAI__', template_text)
                    if srai_match:
                        srai_input = srai_match.group(1)
                        return self._get_response(srai_input, depth + 1)

                # Handle wildcard capture
                response = template_text
                for i, group in enumerate(match.groups()):
                    response = response.replace(f'<star/>', group, 1)
                return response

        return "I'm not sure about that. Type 'help' to see what I can assist with! 🚦"

    def respond(self, user_input: str) -> str:
        """Public method: get bot response to user input"""
        if not user_input or not user_input.strip():
            return "Please ask me something about traffic! 🚦"

        # Add time-aware context
        hour = datetime.now().hour
        if any(word in user_input.lower() for word in ['now', 'current', 'right now', 'currently']):
            if 7 <= hour <= 10:
                suffix = "\n\n⚠️ It's currently morning rush hour (7-10 AM). Expect heavy congestion on major roads!"
            elif 17 <= hour <= 20:
                suffix = "\n\n⚠️ It's currently evening rush hour (5-8 PM). Roads are at peak congestion!"
            else:
                suffix = f"\n\n🕐 Current time: {datetime.now().strftime('%I:%M %p')} — Check the dashboard for live predictions."
        else:
            suffix = ""

        response = self._get_response(user_input)
        return response + suffix


class TrafficChatbot:
    """High-level chatbot interface with conversation history"""

    def __init__(self, aiml_path: str = None):
        if aiml_path is None:
            aiml_path = os.path.join(os.path.dirname(__file__), 'traffic_bot.aiml')
        self.engine = AIMLEngine(aiml_path)
        self.history = []

    def chat(self, user_message: str) -> dict:
        """Process a message and return structured response"""
        response = self.engine.respond(user_message)
        
        entry = {
            "user": user_message,
            "bot": response,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)

        return {
            "response": response,
            "timestamp": entry["timestamp"],
            "history_length": len(self.history)
        }

    def get_history(self) -> list:
        return self.history

    def clear_history(self):
        self.history = []
