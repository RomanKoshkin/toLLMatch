import logging

logger = logging.getLogger("socketio_server_pubsub")


def get_transcoder_output_events(transcoder) -> list:
    speech_and_text_output = transcoder.get_buffered_output()
    if speech_and_text_output is None:
        logger.debug("No output from transcoder.get_buffered_output()")
        return []

    logger.debug(f"We DID get output from the transcoder! {speech_and_text_output}")

    lat = None

    events = []

    if speech_and_text_output.speech_samples:
        events.append(
            {
                "event": "translation_speech",
                "payload": speech_and_text_output.speech_samples,
                "sample_rate": speech_and_text_output.speech_sample_rate,
            }
        )

    if speech_and_text_output.text:
        events.append(
            {
                "event": "translation_text",
                "payload": speech_and_text_output.text,
            }
        )

    for e in events:
        e["eos"] = speech_and_text_output.final

    # if not latency_sent:
    #     lat = transcoder.first_translation_time()
    #     latency_sent = True
    #     to_send["latency"] = lat

    return events
