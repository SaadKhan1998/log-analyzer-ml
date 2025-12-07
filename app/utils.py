import re

def extract_events(log_text: str):
    """
    Simple regex-based event extraction.
    Customize depending on log format.
    """
    lines = log_text.split("\n")
    events = []
    regex = r"\[(.*?)\]\s+(\w+):\s+(.*)"
    
    for line in lines:
        m = re.match(regex, line)
        if m:
            events.append({
                "timestamp": m.group(1),
                "level": m.group(2),
                "message": m.group(3)
            })
    return events
