"""
InvoiceChaser - an AI assistant that writes overdue invoice payment reminders.

Backend only (no UI). Run it in Google Colab or in a normal terminal.

How it works:
  1. Python asks you for the invoice details (one question at a time).
  2. Python calculates the days overdue and suggests a tone.
  3. The AI writes the reminder using ONLY those details.
  4. You can chat to edit it: "make it shorter", "change to Firm", etc.

Setup:
  Colab  -> run  !pip install groq python-dotenv  in a cell first. Key is read
            from the Colab secret named GKey (same as your practice notebook).
  Local  -> install requirements, add GROQ_API_KEY to .env, then run this file.
"""

import os
from datetime import date, datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# 1. SETUP  

def get_api_key():
    """Get the Groq key from the environment, .env file, or Colab secrets."""
    key = os.environ.get("GROQ_API_KEY")
    # if key:
    #     return key

    # try:
    #     from google.colab import userdata
    # except ImportError:
    #     raise ValueError(
    #         "No API key found. Add GROQ_API_KEY to your .env file "
    #         "or set the GROQ_API_KEY environment variable."
    #     ) from None

    # key = userdata.get("GKey")
    # if not key:
    #     raise ValueError("No API key found. Add the GKey secret in Google Colab.")
    return key


MODEL = "openai/gpt-oss-120b"

# How many recent chat messages to send to the model (keep this number EVEN).
MAX_MESSAGES = 8

TONES = ["Friendly", "Professional", "Firm", "Final Notice"]

# Words the user can type to choose a tone.
TONE_ALIASES = {
    "1": "Friendly", "friendly": "Friendly",
    "2": "Professional", "professional": "Professional",
    "3": "Firm", "firm": "Firm",
    "4": "Final Notice", "final": "Final Notice", "final notice": "Final Notice",
}

# Date formats we accept. Note: 05/10/2026 means 5 October 2026 (day first).
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]

# The invoice fields, in the order we ask for them: (key, label).
FIELDS = [
    ("client_name", "Client name"),
    ("invoice_number", "Invoice number"),
    ("amount", "Amount"),
    ("due_date", "Due date"),
    ("payment_instructions", "Payment instructions"),
    ("sender_name", "Sender name"),
    ("extra_notes", "Extra notes"),
]


# 2. SYSTEM PROMPT


SYSTEM = """You are InvoiceChaser, an assistant that writes payment reminder messages for freelancers and small business owners.

YOUR JOB
Write clear, polite and effective reminders for unpaid invoices, and help the user edit them. You will receive the invoice details, a tone, and the status (how many days overdue). You write the reminder text. You do not send it.

INPUT FORMAT
The invoice details may include: client name, invoice number, amount, due date, payment instructions, sender name, extra notes, status and tone. Any detail that was not given is marked NOT PROVIDED.

THE FOUR TONES
- Friendly: warm and relaxed. Assume the client simply forgot. Short and light.
- Professional: polite and businesslike. Clearly state the invoice and ask for payment. Neutral and respectful.
- Firm: serious and direct. Say clearly that the invoice is overdue and ask for payment or a payment date. No anger, no sarcasm.
- Final Notice: the most serious level. Say this is the final reminder, and ask for payment immediately or by the deadline the user gave. Calm, formal and direct.

ADJUST FOR DAYS OVERDUE
The tone sets the voice. The days overdue set the urgency inside that tone.
- Not overdue yet or due today: write a gentle heads-up. Do not say "overdue" or "late".
- 1 to 7 days overdue: assume an honest oversight.
- 8 to 30 days overdue: mention the exact number of days past the due date.
- 31 to 60 days overdue: mention the long delay and ask for payment or a clear payment date.
- More than 60 days overdue: stress the long delay and ask for action now.
- Days overdue UNKNOWN: do not mention how many days late it is.
Use the exact number given. Never calculate or guess days yourself.

FACT RULES (VERY IMPORTANT)
- Use ONLY the details provided. Never invent or guess: names, invoice numbers, amounts, dates, bank or payment details, phone numbers, links, late fees, interest, deadlines, or past reminders.
- Copy names, invoice numbers and amounts exactly as given.
- If a detail is NOT PROVIDED, put a clear placeholder in square brackets, such as [Client name], [Invoice number], [Amount], [Due date], [Payment instructions] or [Your name]. Never skip it silently and never fill it in yourself.
- If anything is missing, add a short note after the reminder listing what is missing and asking the user to send it.
- Mention late fees, interest, deadlines or consequences ONLY if the user gave them in the extra notes, and use their exact terms. For a Final Notice with no deadline given, use the placeholder [payment deadline] and tell the user.
- Do not mention legal action, collection agencies, credit reports or stopping work unless the user wrote it in the extra notes.
- Never threaten violence or public shaming, never contact family or employers, and never insult or guilt-trip the client. Never lie.
- If the user asks you to invent a fee, a threat or a fact, politely decline in one sentence and offer an honest alternative.
- If the user corrects a detail or adds a new one during the chat, use the new detail from then on.
- Do not say the client was contacted before unless the user said so.

EDITING
- When the user asks to shorten, lengthen, change the tone, or change the wording, edit the most recent reminder and give the full updated reminder.
- Keep all invoice details unchanged when editing.
- When shortening, always keep the invoice number, amount, due date, the payment request and the payment instructions.
- When the tone changes, rewrite fully in the new tone and keep the days-overdue rules.
- If the request is unclear, make your best attempt. Ask one short question only if you truly cannot continue.
- If the user wants a WhatsApp or text message, remove the subject line and keep it short.

LIMITS
- You only help with invoice payment reminders and edits to them. For any other topic, politely decline in one sentence and offer to help with the reminder.
- You do not give legal, tax or financial advice. If asked (for example about charging interest or taking legal action), say they should check local rules or ask a qualified professional.
- You cannot send messages, check if an invoice was paid, or look anything up.

RESPONSE STYLE
- Plain text only. No markdown: no asterisks, no hashtags, no tables, no bullet points.
- Simple, clear English. Short paragraphs. A reminder is usually 60 to 130 words (Friendly can be shorter).
- Start directly with the reminder. Do not add an introduction like "Here is your reminder".
- Use this format:
Subject: <short subject line>

<message body>

- Greet the client by name when given. Sign with the sender name when given, otherwise [Your name].
- If you need to add notes (missing details or a short tip), put them after a line containing only three dashes (---).
- No emojis unless the user asks.
"""


# 3. CHAT FUNCTIONS


def get_reply(messages, temperature=0.4, max_tokens=1500):
    """
    Sends chat messages to the Groq model and returns the response content.
    temperature is low (0.4) so the facts stay steady.
    max_tokens is high (1500) so longer replies are not cut off.
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "No API key found. Add GROQ_API_KEY to your .env file "
            "or set the GROQ_API_KEY environment variable."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )

    return response.choices[0].message.content


def trimmed(history):
    """
    Keep the conversation short, but NEVER lose the invoice details.

    history[0] = system prompt, history[1] = invoice details message.
    Both are always kept (system FIRST - this fixes the order from the notebook).
    Then we add only the most recent messages.
    Because MAX_MESSAGES is even, the recent part always starts with the
    assistant, so the roles stay in the correct order (user, assistant, user...).
    """
    start = history[:2]
    recent = history[2:][-MAX_MESSAGES:]
    return start + recent


def chat(history, user_text):
    """
    Add the user's message, get the reply, and save the reply in history.
    Returns the reply text, or None if something went wrong.
    """
    history.append({"role": "user", "content": user_text})

    try:
        reply = get_reply(trimmed(history))
    except Exception as error:      # network problem, wrong key, rate limit...
        history.pop()               # remove the message that failed
        print(f"bot: Sorry, I could not reach the AI service ({error}).")
        return None

    if not reply or not reply.strip():
        history.pop()
        print("bot: I got an empty answer. Please try again.")
        return None

    history.append({"role": "assistant", "content": reply.strip()})
    return reply.strip()


def show_reply(reply):
    """Print a reply between lines so it is easy to copy."""
    print("\n" + "=" * 60)
    print(reply)
    print("=" * 60 + "\n")


# 4. INVOICE LOGIC  

class QuitChat(Exception):
    """Raised when the user types quit. Used to leave the program cleanly."""


def ask(prompt):
    """Ask the user a question. Typing quit or exit stops the program."""
    text = input(prompt).strip()
    if text.lower() in ("quit", "exit"):
        raise QuitChat
    return text


def ask_optional(prompt):
    """Like ask(), but pressing Enter returns None (meaning: not provided)."""
    text = ask(prompt)
    return text if text else None


def parse_date(text):
    """Turn text into a date. Returns None if the text is not a valid date."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def ask_due_date():
    """Keep asking until we get a valid date, or the user skips it."""
    while True:
        text = ask("Due date (YYYY-MM-DD or DD/MM/YYYY, Enter to skip): ")
        if not text:
            return None
        due = parse_date(text)
        if due:
            return due
        print("bot: I could not read that date. Example: 2026-10-05 or 05/10/2026")


def calculate_days_overdue(due_date, today=None):
    """
    Days between the due date and today.
    Positive = overdue, 0 = due today, negative = not due yet.
    Returns None if there is no due date.
    """
    if due_date is None:
        return None
    if today is None:
        today = date.today()
    return (today - due_date).days


def suggest_tone(days_overdue):
    """Pick a default tone from the number of days overdue. Change the numbers if you like."""
    if days_overdue is None:
        return "Professional"
    if days_overdue <= 7:
        return "Friendly"
    if days_overdue <= 30:
        return "Professional"
    if days_overdue <= 60:
        return "Firm"
    return "Final Notice"


def ask_tone(suggested):
    """Let the user pick a tone. Pressing Enter accepts the suggested one."""
    while True:
        text = ask(f"Tone: 1 Friendly, 2 Professional, 3 Firm, 4 Final Notice (Enter = {suggested}): ").lower()
        if not text:
            return suggested
        if text in TONE_ALIASES:
            return TONE_ALIASES[text]
        print("bot: Please type 1, 2, 3, 4 or a tone name.")


def collect_invoice_details():
    """Ask for each invoice detail. Pressing Enter skips a detail (stored as None)."""
    details = {}
    details["client_name"] = ask_optional("Client name: ")
    details["invoice_number"] = ask_optional("Invoice number: ")
    details["amount"] = ask_optional("Amount with currency (e.g. $500 or PKR 45,000): ")
    details["due_date"] = ask_due_date()
    details["payment_instructions"] = ask_optional("Payment instructions (bank details, payment link, etc.): ")
    details["sender_name"] = ask_optional("Your name or business name (to sign the message): ")
    details["extra_notes"] = ask_optional("Extra notes, e.g. late fee terms or a deadline (optional): ")
    return details


def get_missing(details):
    """List the labels of important details that were skipped."""
    missing = []
    for key, label in FIELDS:
        if key != "extra_notes" and details[key] is None:    # extra notes are truly optional
            missing.append(label)
    return missing


def plural_days(number):
    """1 -> '1 day', 5 -> '5 days'"""
    return f"{number} day" if number == 1 else f"{number} days"


def describe_days(days_overdue):
    """Turn the number of days into a short sentence for the AI (and for the user)."""
    if days_overdue is None:
        return "UNKNOWN (due date not provided)"
    if days_overdue > 0:
        return f"{plural_days(days_overdue)} overdue"
    if days_overdue == 0:
        return "due today (not overdue yet)"
    return f"not overdue yet (due in {plural_days(-days_overdue)})"


def build_facts_message(details, days_overdue, tone):
    """
    Build the first message for the AI. It lists every fact clearly.
    Anything the user skipped is written as NOT PROVIDED, so the AI will
    use a placeholder instead of inventing a value.
    """
    lines = ["Write a payment reminder using ONLY the invoice details below.", ""]

    for key, label in FIELDS:
        value = details[key]
        if key == "due_date" and value is not None:
            value = value.strftime("%d %B %Y")       # e.g. 05 October 2026
        if value is None:
            value = "NOT PROVIDED"
        lines.append(f"{label}: {value}")

    missing = get_missing(details)
    lines.append(f"Status: {describe_days(days_overdue)}")
    lines.append(f"Tone: {tone}")
    lines.append("Missing details: " + (", ".join(missing) if missing else "none"))
    return "\n".join(lines)


# 5. THE CHAT LOOP

HELP_TEXT = """You can now ask for changes, for example:
  - make it shorter
  - change the tone to Firm
  - add that I accept bank transfer
  - turn it into a WhatsApp message
Type /new for another invoice, or quit to exit."""


def run_invoice_session():
    """One full invoice: collect details -> first reminder -> edit loop."""
    print("\n--- New invoice --- (press Enter to skip a detail, type quit to exit)\n")

    details = collect_invoice_details()

    days_overdue = calculate_days_overdue(details["due_date"])
    suggested = suggest_tone(days_overdue)
    print(f"\nbot: Status: {describe_days(days_overdue)}. Suggested tone: {suggested}.")
    tone = ask_tone(suggested)

    # Small warning if the chosen tone is much stronger than suggested.
    if TONES.index(tone) - TONES.index(suggested) >= 2:
        print("bot: Tip: that tone is much stronger than usual for this stage. It may upset the client.")

    missing = get_missing(details)
    if missing:
        print("bot: Missing: " + ", ".join(missing) + ". I will use [placeholders] for these.")

    facts = build_facts_message(details, days_overdue, tone)

    # Each invoice starts with a fresh history (system prompt first).
    history = [{"role": "system", "content": SYSTEM}]

    print("\nbot: Writing your reminder...")
    reply = chat(history, facts)
    while reply is None:                    
        ask("Press Enter to try again (or type quit to exit): ")
        reply = chat(history, facts)

    show_reply(reply)
    print(HELP_TEXT)

    while True:
        user_text = ask("\nYou: ")
        if not user_text:
            continue
        if user_text.lower() == "/new":
            return                        

        reply = chat(history, user_text)
        if reply is not None:
            show_reply(reply)


def main():
    print("InvoiceChaser - overdue invoice reminder assistant")
    try:
        while True:
            run_invoice_session()
    except (QuitChat, EOFError):
        print("\nbot: Goodbye! I hope your invoices get paid soon.")


if __name__ == "__main__":
    main()