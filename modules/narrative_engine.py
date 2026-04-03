"""
Module 9 — Narrative Engine
Generates context-aware, proactive narrations from inferred social state.
Prevents repetitive name-calling and creates a more "assistant-like" feel.
"""

import random
import time
import config


class NarrativeEngine:
    """
    Decides WHAT to say and WHEN to say it based on ContextEngine output.
    """

    def __init__(self, style="proactive", summary_cooldown=30.0, llm_engine=None):
        """
        Parameters
        ----------
        style : str
            'proactive' (narrates vibes) or 'reactive' (focused on detections).
        summary_cooldown : float
            Seconds between room vibe summaries.
        llm_engine : LLMEngine | None
            Optional LLM backend for natural language generation.
        """
        self.style = style
        self.summary_cooldown = summary_cooldown
        self.llm_engine = llm_engine
        self.last_summary_time = 0
        self.last_llm_call_time = 0
        
        # Keep track of what we just said to avoid repetition
        self.last_narrated_entities = set()
        self.last_vibe = None
        self.recently_introduced = {} # {name: timestamp}

    def generate_narration(self, persons, context_summary, interactions):
        """
        Synthesizes a narration string.
        
        Returns
        -------
        str | None
            The text to speak, or None if no important update.
        """
        now = time.time()
        vibe = context_summary["vibe"]

        # 1. LLM Brain (Proactive Social Context)
        if self.llm_engine and self.llm_engine.enabled:
            # Check LLM Rate Limit Cooldown
            time_since_llm = now - self.last_llm_call_time
            
            # Urgent Trigger: Overrides Cooldown (e.g., someone is waving or stopping)
            urgent_interaction = any(i in str(interactions).lower() for i in ["stop", "wave", "point", "handshake"])
            
            if time_since_llm < config.LLM_COOLDOWN and not urgent_interaction:
                # LLM is sleeping to respect rate limits. Fall back to heuristic or silence.
                return None

            self.last_llm_call_time = now
            # Clean up old social memory (120s)
            self.recently_introduced = {n: t for n, t in self.recently_introduced.items() if now - t < 120}

            llm_context = {
                "global_vibe": vibe,
                "people_present": [
                    {
                        "name": p["identity"]["name"],
                        "emotion": p["emotion"]["emotion"],
                        "gesture": p["gesture"],
                        "already_introduced": p["identity"]["name"] in self.recently_introduced
                    } for p in persons
                ],
                "social_interactions": interactions,
                "timestamp": time.strftime("%H:%M:%S"),
                "seconds_since_last_spoken": int(now - self.last_summary_time)
            }
            
            decision = self.llm_engine.generate_narrative_decision(llm_context)
            
            if decision:
                importance = decision.get("importance", 0)
                should_speak = decision.get("should_speak", False)
                message = decision.get("message", "")
                
                # Check for Urgency Override
                is_urgent = importance >= config.LLM_URGENCY_THRESHOLD
                
                if should_speak and message:
                    if is_urgent or (now - self.last_summary_time > self.summary_cooldown):
                        self.last_summary_time = now
                        self.last_vibe = vibe
                        
                        # Update Social Memory
                        for p in persons:
                            name = p["identity"]["name"]
                            if name != "Unknown":
                                self.recently_introduced[name] = now

                        return message
                    
            # If LLM is active but decided NOT to speak (or wasn't urgent enough to break cooldown),
            # return None to enforce silence.
            return None

        # 2. Fallback to Heuristic Phrases (If LLM is disabled)
        should_summarize = (now - self.last_summary_time > self.summary_cooldown) or (vibe != self.last_vibe)
        if not should_summarize or context_summary["people_total"] == 0:
            return None

        narrative = []
        msg = self._summarize_vibe(context_summary)
        if msg:
            narrative.append(msg)
        
        for interaction in interactions:
            narrative.append(f"Interesting; {interaction}.")

        # 3. Add specific person alerts only if new or changing significantly
        # (Handling this via the existing cooldown in main.py for now, 
        # but the NarrativeEngine could eventually take over full identity management)

        if not narrative:
            return None
            
        self.last_summary_time = now
        self.last_vibe = vibe
        return " ".join(narrative)

    def _summarize_vibe(self, context):
        """
        Choose a natural phrasing for the current room state.
        """
        vibe = context["vibe"]
        total = context["people_total"]
        known = context["known_count"]
        
        if total == 0:
            return None

        phrases = {
            "Positive": [
                "The atmosphere here is wonderful; everyone seems happy.",
                "It's a very positive vibe in the room right now.",
                "Lots of smiles! It feels great in here."
            ],
            "Tense": [
                "I sense some tension in the room.",
                "Things seem a bit intense right now.",
                "A bit of a tense atmosphere."
            ],
            "Heavy": [
                "The room feels a bit somber.",
                "A somewhat low-energy or sad vibe in here."
            ],
            "Calm": [
                "Everything is quite calm and relaxed.",
                "It's a peaceful atmosphere.",
                "Things are steady and quiet."
            ]
        }

        # Select a phrase
        base = random.choice(phrases.get(vibe, phrases["Calm"]))
        
        # Append population context
        if known == total:
            counts = f" It's just you and the people I know." if total > 1 else ""
        elif known > 0:
            counts = f" I see {known} familiar faces and {total - known} new people."
        else:
            counts = f" I'm seeing {total} people I don't recognize yet."

        return base + counts
