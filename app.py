from datetime import date

import streamlit as st

import main as invoice_chaser


st.set_page_config(page_title="InvoiceChaser", page_icon="💬")
st.title("InvoiceChaser")
st.caption("Create a polite, factual reminder for an unpaid invoice.")

DETAIL_KEYS = (
    "client_name",
    "invoice_number",
    "amount",
    "due_date",
    "payment_instructions",
    "sender_name",
    "extra_notes",
)


def start_new_invoice():
    st.session_state.pop("conversation", None)
    for key in (*DETAIL_KEYS, "has_due_date", "tone"):
        st.session_state.pop(key, None)


if "conversation" not in st.session_state:
    st.session_state.conversation = None

with st.sidebar:
    st.header("Invoice details")
    st.text_input("Client name", key="client_name")
    st.text_input("Invoice number", key="invoice_number")
    st.text_input("Amount with currency", placeholder="e.g. $500 or PKR 45,000", key="amount")

    st.checkbox("I know the due date", value=False, key="has_due_date")
    if st.session_state.has_due_date:
        st.date_input("Due date", value=date.today(), key="due_date")

    st.text_area(
        "Payment instructions",
        placeholder="Bank details, payment link, or other instructions",
        key="payment_instructions",
    )
    st.text_input("Your name or business name", key="sender_name")
    st.text_area(
        "Extra notes",
        placeholder="Optional: include only terms or deadlines you want mentioned",
        key="extra_notes",
    )

    due_date = st.session_state.get("due_date") if st.session_state.has_due_date else None
    days_overdue = invoice_chaser.calculate_days_overdue(due_date)
    suggested_tone = invoice_chaser.suggest_tone(days_overdue)
    st.caption(f"Status: {invoice_chaser.describe_days(days_overdue)}")
    tone = st.selectbox(
        "Reminder tone",
        invoice_chaser.TONES,
        index=invoice_chaser.TONES.index(suggested_tone),
        key="tone",
    )

    generate_clicked = st.button("Generate reminder", type="primary", use_container_width=True)
    st.button("New invoice", on_click=start_new_invoice, use_container_width=True)

if generate_clicked:
    details = {
        "client_name": st.session_state.client_name.strip() or None,
        "invoice_number": st.session_state.invoice_number.strip() or None,
        "amount": st.session_state.amount.strip() or None,
        "due_date": due_date,
        "payment_instructions": st.session_state.payment_instructions.strip() or None,
        "sender_name": st.session_state.sender_name.strip() or None,
        "extra_notes": st.session_state.extra_notes.strip() or None,
    }
    missing = invoice_chaser.get_missing(details)
    if missing:
        st.warning(
            "Some details are missing. The reminder will use placeholders for: "
            + ", ".join(missing)
        )
    if invoice_chaser.TONES.index(tone) - invoice_chaser.TONES.index(suggested_tone) >= 2:
        st.info("This tone is much stronger than usually suggested for this invoice status.")

    facts = invoice_chaser.build_facts_message(details, days_overdue, tone)
    conversation = [
        {"role": "system", "content": invoice_chaser.SYSTEM},
        {"role": "user", "content": facts},
    ]
    try:
        reply = invoice_chaser.get_reply(invoice_chaser.trimmed(conversation))
        if not reply or not reply.strip():
            raise ValueError("The AI returned an empty response. Please try again.")
        conversation.append({"role": "assistant", "content": reply.strip()})
        st.session_state.conversation = conversation
    except Exception as error:
        st.error(f"Could not generate the reminder: {error}")

conversation = st.session_state.conversation
if conversation:
    st.subheader("Reminder")
    for message in conversation[2:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    latest_reminder = next(
        (message["content"] for message in reversed(conversation) if message["role"] == "assistant"),
        None,
    )
    if latest_reminder:
        st.download_button(
            "Download reminder",
            data=latest_reminder,
            file_name="invoice-reminder.txt",
            mime="text/plain",
        )

    user_text = st.chat_input("Ask to edit the reminder, e.g. “make it shorter”")
    if user_text:
        conversation.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)
        try:
            reply = invoice_chaser.get_reply(invoice_chaser.trimmed(conversation))
            if not reply or not reply.strip():
                raise ValueError("The AI returned an empty response. Please try again.")
            conversation.append({"role": "assistant", "content": reply.strip()})
            st.rerun()
        except Exception as error:
            conversation.pop()
            st.error(f"Could not edit the reminder: {error}")