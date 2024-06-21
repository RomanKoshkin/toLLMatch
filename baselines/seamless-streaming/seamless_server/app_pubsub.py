from operator import itemgetter
import os
from typing import Any, Optional, Tuple, Dict, TypedDict
from urllib import parse
from uuid import uuid4
import colorlog
import io
import logging
from pprint import pformat
import socketio
import sys
import time
import random
import string
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles


from src.room import Room, Member
from src.simuleval_agent_directory import NoAvailableAgentException
from src.simuleval_agent_directory import SimulevalAgentDirectory
from src.simuleval_transcoder import SimulevalTranscoder
from src.transcoder_helpers import get_transcoder_output_events

###############################################
# Constants
###############################################

DEBUG = True

ALL_ROOM_ID = "ALL"

ROOM_ID_USABLE_CHARACTERS = string.ascii_uppercase
ROOM_ID_LENGTH = 4

ROOM_LISTENERS_SUFFIX = "_listeners"
ROOM_SPEAKERS_SUFFIX = "_speakers"

ESCAPE_HATCH_SERVER_LOCK_RELEASE_NAME = "remove_server_lock"

###############################################
# Configure logger
###############################################

logger = logging.getLogger("socketio_server_pubsub")
logger.propagate = False

handler = colorlog.StreamHandler(stream=sys.stdout)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(levelname)s][%(module)s]:%(reset)s %(message)s",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.WARNING)

print("")
print("")
print("=" * 20 + " ⭐️ Starting Server... ⭐️ " + "=" * 20)

###############################################
# Configure socketio server
###############################################

CLIENT_BUILD_PATH = "../streaming-react-app/dist/"
static_files = {
    "/": CLIENT_BUILD_PATH,
    "/assets/seamless-db6a2555.svg": {
        "filename": CLIENT_BUILD_PATH + "assets/seamless-db6a2555.svg",
        "content_type": "image/svg+xml",
    },
}

# sio is the main socket.io entrypoint
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=logger,
    # engineio_logger=logger,
)
# sio.logger.setLevel(logging.DEBUG)
socketio_app = socketio.ASGIApp(sio)

app_routes = [
    Mount("/ws", app=socketio_app),  # Mount Socket.IO server under /app
    Mount(
        "/", app=StaticFiles(directory=CLIENT_BUILD_PATH, html=True)
    ),  # Serve static files from root
]
app = Starlette(debug=True, routes=app_routes)

# rooms is indexed by room_id
rooms: Dict[str, Room] = {}


class MemberDirectoryObject(TypedDict):
    room: Room
    member_object: Member


# member_directory is indexed by client_id
# NOTE: client_id is really "client session id", meaning that it is unique to a single browser session.
# If a user opens a new tab, they will have a different client_id and can join another room, join
# the same room with different roles, etc.
# NOTE: For a long-running production server we would want to clean up members after a certain timeout
# but for this limited application we can just keep them around
member_directory: Dict[str, MemberDirectoryObject] = {}


class ServerLock(TypedDict):
    name: str
    client_id: str
    member_object: Member

MAX_SPEAKERS = os.environ.get("MAX_SPEAKERS")

if os.environ.get("LOCK_SERVER_COMPLETELY", "0") == "1":
    logger.info("LOCK_SERVER_COMPLETELY is set. Server will be locked on startup.")
if MAX_SPEAKERS is not None and int(MAX_SPEAKERS):
    logger.info(f"MAX_SPEAKERS is set to: {MAX_SPEAKERS}")
dummy_server_lock_member_object = Member(
    client_id="seamless_user", session_id="dummy", name="Seamless User"
)
# Normally this would be an actual transcoder, but it's fine putting True here since currently we only check for the presence of the transcoder
dummy_server_lock_member_object.transcoder = True
server_lock: Optional[ServerLock] = (
    {
        "name": "Seamless User",
        "client_id": "seamless_user",
        "member_object": dummy_server_lock_member_object,
    }
    if os.environ.get("LOCK_SERVER_COMPLETELY", "0") == "1"
    else None
)

server_id = str(uuid4())

# Specify specific models to load (some environments have issues loading multiple models)
# See AgentWithInfo with JSON format details.
models_override = os.environ.get("MODELS_OVERRIDE")

available_agents = SimulevalAgentDirectory()
logger.info("Building and adding agents...")
if models_override is not None:
    logger.info(f"MODELS_OVERRIDE supplied from env vars: {models_override}")
available_agents.build_and_add_agents(models_override)

agents_capabilities_for_json = available_agents.get_agents_capabilities_list_for_json()


###############################################
# Helpers
###############################################


def catch_and_log_exceptions_for_sio_event_handlers(func):
    # wrapper should have the same signature as the original function
    async def catch_exception_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            message = f"[app_pubsub] Caught exception in '{func.__name__}' event handler:\n\n{e}"
            logger.exception(message, stack_info=True)

            try:
                exception_data = {
                    "message": message,
                    "timeEpochMs": int(time.time() * 1000),
                }

                try:
                    # Let's try to add as much useful metadata as possible to the server_exception event
                    sid = args[0]
                    if isinstance(sid, str) and len(sid) > 0:
                        session_data = await get_session_data(sid)
                        if session_data:
                            client_id = session_data.get("client_id")
                            member = session_data.get("member_object")
                            room = session_data.get("room_object")

                            exception_data["room"] = str(room)
                            exception_data["member"] = str(member)
                            exception_data["clientID"] = str(client_id)
                except Exception as inner_e:
                    # We expect there will be times when clientID or other values aren't present, so just log this as a warning
                    logger.warn(
                        f"[app_pubsub] Caught exception while trying add additional_data to server_exception:\n\n{inner_e}"
                    )

                # For now let's emit this to all clients. We ultimatley may want to emit it just to the room it's happening in.
                await sio.emit("server_exception", exception_data)
            except Exception as inner_e:
                logger.exception(
                    f"[app_pubsub] Caught exception while trying to emit server_exception event:\n{inner_e}"
                )

            # Re-raise the exception so it's handled normally by the server
            raise e

    # Set the name of the wrapper to the name of the original function so that the socketio server can associate it with the right event
    catch_exception_wrapper.__name__ = func.__name__
    return catch_exception_wrapper


async def emit_room_state_update(room):
    await sio.emit(
        "room_state_update",
        room.to_json(),
        room=room.room_id,
    )


async def emit_server_state_update():
    room_statuses = {
        room_id: room.get_room_status_dict() for room_id, room in rooms.items()
    }
    total_active_connections = sum(
        [room_status["activeConnections"] for room_status in room_statuses.values()]
    )
    total_active_transcoders = sum(
        [room_status["activeTranscoders"] for room_status in room_statuses.values()]
    )
    logger.info(
        f"[Server Status]: {total_active_connections} active connections (in rooms); {total_active_transcoders} active transcoders"
    )
    logger.info(f"[Server Status]: server_lock={server_lock}")
    server_lock_object_for_js = (
        {
            "name": server_lock.get("name"),
            "clientID": server_lock.get("client_id"),
            "isActive": server_lock.get("member_object")
            and server_lock.get("member_object").transcoder is not None,
        }
        if server_lock
        else None
    )
    await sio.emit(
        "server_state_update",
        {
            "statusByRoom": room_statuses,
            "totalActiveConnections": total_active_connections,
            "totalActiveTranscoders": total_active_transcoders,
            "agentsCapabilities": agents_capabilities_for_json,
            "serverLock": server_lock_object_for_js,
        },
        room=ALL_ROOM_ID,
    )


async def get_session_data(sid):
    session = await sio.get_session(sid)
    # It seems like if the session has not been set that get_session may return None, so let's provide a fallback empty dictionary here
    return session or {}


async def set_session_data(sid, client_id, room_id, room_object, member_object):
    await sio.save_session(
        sid,
        {
            "client_id": client_id,
            "room_id": room_id,
            "room_object": room_object,
            "member_object": member_object,
        },
    )


def get_random_room_id():
    return "".join(random.choices(ROOM_ID_USABLE_CHARACTERS, k=ROOM_ID_LENGTH))


def get_random_unused_room_id():
    room_id = get_random_room_id()
    while room_id in rooms:
        room_id = get_random_room_id()
    return room_id


###############################################
# Socket.io Basic Event Handlers
###############################################


@sio.on("connect")
@catch_and_log_exceptions_for_sio_event_handlers
async def connect(sid, environ):
    logger.info(f"📥 [event: connected] sid={sid}")

    # TODO: Sanitize/validate query param input
    query_params = dict(parse.parse_qsl(environ["QUERY_STRING"]))
    client_id = query_params.get("clientID")

    logger.debug(f"query_params:\n{pformat(query_params)}")

    if client_id is None:
        logger.info("No clientID provided. Disconnecting...")
        await sio.disconnect(sid)
        return

    # On reconnect we need to rejoin rooms and reset session data
    if member_directory.get(client_id):
        room = member_directory[client_id].get("room")
        room_id = room.room_id
        # Note: We could also get this from room.members[client_id]
        member = member_directory[client_id].get("member_object")

        member.connection_status = "connected"
        member.session_id = sid

        logger.info(
            f"[event: connect] {member} reconnected. Attempting to re-add them to socketio rooms and reset session data."
        )

        if room is None or member is None:
            logger.error(
                f"[event: connect] {client_id} is reconnecting, but room or member is None. This should not happen."
            )
            await sio.disconnect(sid)
            return

        sio.enter_room(sid, room_id)
        sio.enter_room(sid, ALL_ROOM_ID)

        if client_id in room.listeners:
            sio.enter_room(sid, f"{room_id}{ROOM_LISTENERS_SUFFIX}")
        if client_id in room.speakers:
            sio.enter_room(sid, f"{room_id}{ROOM_SPEAKERS_SUFFIX}")

        # Save the room_id to the socketio client session
        await set_session_data(
            sid,
            client_id=client_id,
            room_id=room.room_id,
            room_object=room,
            member_object=member,
        )
        await emit_room_state_update(room)
    else:
        # Save the client id to the socketio client session
        await set_session_data(
            sid, client_id=client_id, room_id=None, room_object=None, member_object=None
        )

    await sio.emit("server_id", server_id, to=sid)
    await emit_server_state_update()


@sio.event
@catch_and_log_exceptions_for_sio_event_handlers
async def disconnect(sid):
    global server_lock
    session_data = await get_session_data(sid)
    # logger.info("session_data", session_data)

    client_id = None
    member = None
    room = None

    if session_data:
        client_id = session_data.get("client_id")
        member = session_data.get("member_object")
        room = session_data.get("room_object")

    logger.info(
        f"[event: disconnect][{room or 'NOT_IN_ROOM'}] member: {member or 'NO_MEMBER_OBJECT'} disconnected"
    )

    # Release the lock if this is the client that holds the current server lock
    if server_lock and server_lock.get("client_id") == client_id:
        server_lock = None

    if member:
        member.connection_status = "disconnected"

        if member.transcoder:
            member.transcoder.close = True
            member.transcoder = None
            member.requested_output_type = None

        if room:
            logger.info(
                f"[event: disconnect] {member} disconnected from room {room.room_id}"
            )
            await emit_room_state_update(room)
        else:
            logger.info(
                f"[event: disconnect] {member} disconnected, but no room object present. This should not happen."
            )
    else:
        logger.info(
            f"[event: disconnect] client_id {client_id or 'NO_CLIENT_ID'} with sid {sid} in rooms {str(sio.rooms(sid))} disconnected"
        )

    await emit_server_state_update()


@sio.on("*")
async def catch_all(event, sid, data):
    logger.info(f"[unhandled event: {event}] sid={sid} data={data}")


###############################################
# Socket.io Streaming Event handlers
###############################################


@sio.on("join_room")
@catch_and_log_exceptions_for_sio_event_handlers
async def join_room(sid, client_id, room_id_from_client, config_dict):
    global server_lock

    args = {
        "sid": sid,
        "client_id": client_id,
        "room_id": room_id_from_client,
        "config_dict": config_dict,
    }
    logger.info(f"[event: join_room] {args}")
    session_data = await get_session_data(sid)
    logger.info(f"session_data: {session_data}")

    room_id = room_id_from_client
    if room_id is None:
        room_id = get_random_unused_room_id()
        logger.info(
            f"No room_id provided. Generating a random, unused room_id: {room_id}"
        )

    # Create the room if it doesn't already exist
    if room_id not in rooms:
        rooms[room_id] = Room(room_id)

    room = rooms[room_id]

    member = None

    name = "[NO_NAME]"

    # If the client is reconnecting use their existing member object. Otherwise create a new one.
    if client_id in room.members:
        member = room.members[client_id]
        logger.info(f"{member} is rejoining room {room_id}.")
    else:
        member_number = len(room.members) + 1
        name = f"Member {member_number}"
        member = Member(
            client_id=client_id,
            session_id=sid,
            name=name,
        )
        logger.info(f"Created a new Member object: {member}")
        logger.info(f"Adding {member} to room {room_id}")
        room.members[client_id] = member

    # Also add them to the member directory
    member_directory[client_id] = {"room": room, "member_object": member}

    # Join the socketio room, which enables broadcasting to all members of the room
    sio.enter_room(sid, room_id)
    # Join the room for all clients
    sio.enter_room(sid, ALL_ROOM_ID)

    if "listener" in config_dict["roles"]:
        sio.enter_room(sid, f"{room_id}{ROOM_LISTENERS_SUFFIX}")
        if client_id not in room.listeners:
            room.listeners.append(client_id)
    else:
        sio.leave_room(sid, f"{room_id}{ROOM_LISTENERS_SUFFIX}")
        room.listeners = [
            listener_id for listener_id in room.listeners if listener_id != client_id
        ]

    if "speaker" in config_dict["roles"]:
        sio.enter_room(sid, f"{room_id}{ROOM_SPEAKERS_SUFFIX}")
        if client_id not in room.speakers:
            room.speakers.append(client_id)
    else:
        sio.leave_room(sid, f"{room_id}{ROOM_SPEAKERS_SUFFIX}")
        # If the person is no longer a speaker they should no longer be able to lock the server
        if server_lock and server_lock.get("client_id") == client_id:
            logger.info(
                f"🔓 Server is now unlocked from client {server_lock.get('client_id')} with name/info: {server_lock.get('name')}"
            )
            server_lock = None
        if member.transcoder:
            member.transcoder.close = True
            member.transcoder = None
        room.speakers = [
            speaker_id for speaker_id in room.speakers if speaker_id != client_id
        ]

    # If we currently own the server lock and are updating roles and we no longer have server lock specified, release it
    if (
        server_lock is not None
        and server_lock["client_id"] == client_id
        and config_dict.get("lockServerName") is None
    ):
        logger.info(f"[join_room] Releasing server lock: {pformat(server_lock)}")
        server_lock = None

    # Only speakers should be able to lock the server
    if config_dict.get("lockServerName") is not None and "speaker" in config_dict.get(
        "roles", {}
    ):
        # If something goes wrong and the server gets stuck in a locked state the client can
        # force the server to remove the lock by passing the special name ESCAPE_HATCH_SERVER_LOCK_RELEASE_NAME
        if (
            server_lock is not None
            and config_dict.get("lockServerName")
            == ESCAPE_HATCH_SERVER_LOCK_RELEASE_NAME
            # If we are locking the server completely we don't want someone to be able to unlock it
            and not os.environ.get("LOCK_SERVER_COMPLETELY", "0") == "1"
        ):
            server_lock = None
            logger.info(
                f"🔓 Server lock has been reset by {client_id} using the escape hatch name {ESCAPE_HATCH_SERVER_LOCK_RELEASE_NAME}"
            )

        # If the server is not locked, set a lock. If it's already locked to this client, update the lock object
        if server_lock is None or server_lock.get("client_id") == client_id:
            # TODO: Add some sort of timeout as a backstop in case someone leaves the browser tab open after locking the server
            server_lock = {
                "name": config_dict.get("lockServerName"),
                "client_id": client_id,
                "member_object": member,
            }
            logger.info(
                f"🔒 Server is now locked to client {server_lock.get('client_id')} with name/info: {server_lock.get('name')}\nThis client will have priority over all others until they disconnect."
            )
        # If the server is already locked to someone else, don't allow this client to lock it
        elif server_lock is not None and server_lock.get("client_id") != client_id:
            logger.warn(
                f"⚠️  Server is already locked to client {server_lock.get('client_id')}. Ignoring request to lock to client {client_id}."
            )
            # TODO: Maybe throw an error here?

    # Save the room_id to the socketio client session
    await set_session_data(
        sid,
        client_id=client_id,
        room_id=room_id,
        room_object=room,
        member_object=member,
    )

    await emit_room_state_update(room)
    await emit_server_state_update()

    return {"roomsJoined": sio.rooms(sid), "roomID": room_id}

def allow_speaker(room, client_id):
    if MAX_SPEAKERS is not None and client_id in room.speakers:
        room_statuses = {room_id: room.get_room_status_dict() for room_id, room in rooms.items()}
        speakers = sum(room_status["activeTranscoders"] for room_status in room_statuses.values())
        return speakers < int(MAX_SPEAKERS)
    return True

# TODO: Add code to prevent more than one speaker from connecting/streaming at a time
@sio.event
@catch_and_log_exceptions_for_sio_event_handlers
async def configure_stream(sid, config):
    session_data = await get_session_data(sid)
    client_id, member, room = itemgetter("client_id", "member_object", "room_object")(
        session_data
    )

    logger.debug(
        f"[event: configure_stream][{room}] Received stream config from {member}\n{pformat(config)}"
    )

    if member is None or room is None:
        logger.error(
            f"Received stream config from {member}, but member or room is None. This should not happen."
        )
        return {"status": "error", "message": "member_or_room_is_none"}

    if not allow_speaker(room, client_id):
        logger.error(
            f"In MAX_SPEAKERS mode we only allow one speaker at a time. Ignoring request to configure stream from client {client_id}."
        )
        return {"status": "error", "message": "max_speakers"}

    # If there is a server lock WITH an active transcoder session, prevent other users from configuring and starting a stream
    # If the server lock client does NOT have an active transcoder session allow this to proceed, knowing that
    # this stream will be interrupted if the server lock client starts streaming
    if (
        server_lock is not None
        and server_lock.get("client_id") != client_id
        and server_lock.get("member_object")
        and server_lock.get("member_object").transcoder is not None
    ):
        logger.warn(
            f"Server is locked to client {server_lock.get('client_id')}. Ignoring request to configure stream from client {client_id}."
        )
        return {"status": "error", "message": "server_locked"}

    debug = config.get("debug")
    async_processing = config.get("async_processing")

    # Currently s2s, s2t or s2s&t
    model_type = config.get("model_type")
    member.requested_output_type = model_type

    model_name = config.get("model_name")

    try:
        agent = available_agents.get_agent_or_throw(model_name)
    except NoAvailableAgentException as e:
        logger.warn(f"Error while getting agent: {e}")
        # await sio.emit("error", str(e), to=sid)
        await sio.disconnect(sid)
        return {"status": "error", "message": str(e)}

    if member.transcoder:
        logger.warn(
            "Member already has a transcoder configured. Closing it, and overwriting with a new transcoder..."
        )
        member.transcoder.close = True

    t0 = time.time()
    try:
        member.transcoder = SimulevalTranscoder(
            agent,
            config["rate"],
            debug=debug,
            buffer_limit=int(config["buffer_limit"]),
        )
    except Exception as e:
        logger.warn(f"Got exception while initializing agents: {e}")
        # await sio.emit("error", str(e), to=sid)
        await sio.disconnect(sid)
        return {"status": "error", "message": str(e)}

    t1 = time.time()
    logger.debug(f"Booting up VAD and transcoder took {t1-t0} sec")

    # TODO: if async_processing is false, then we need to run transcoder.process_pipeline_once() whenever we receive audio, or at some other sensible interval
    if async_processing:
        member.transcoder.start()

    # We need to emit a room state update here since room state now includes # of active transcoders
    await emit_room_state_update(room)
    await emit_server_state_update()

    return {"status": "ok", "message": "server_ready"}


# The config here is a partial config, meaning it may not contain all the config values -- only the ones the user
# wants to change
@sio.on("set_dynamic_config")
@catch_and_log_exceptions_for_sio_event_handlers
async def set_dynamic_config(
    sid,
    # partial_config's type is defined in StreamingTypes.ts
    partial_config,
):
    session_data = await get_session_data(sid)

    member = None

    if session_data:
        member = session_data.get("member_object")

    if member:
        new_dynamic_config = {
            **(member.transcoder_dynamic_config or {}),
            **partial_config,
        }
        logger.info(
            f"[set_dynamic_config] Setting new dynamic config:\n\n{pformat(new_dynamic_config)}\n"
        )
        member.transcoder_dynamic_config = new_dynamic_config

    return {"status": "ok", "message": "dynamic_config_set"}


@sio.event
@catch_and_log_exceptions_for_sio_event_handlers
async def incoming_audio(sid, blob):
    session_data = await get_session_data(sid)

    client_id = None
    member = None
    room = None

    if session_data:
        client_id = session_data.get("client_id")
        member = session_data.get("member_object")
        room = session_data.get("room_object")

    logger.debug(f"[event: incoming_audio] from member {member}")

    # If the server is locked by someone else, kill our transcoder and ignore incoming audio
    # If the server lock client does NOT have an active transcoder session allow this incoming audio pipeline to proceed,
    # knowing that this stream will be interrupted if the server lock client starts streaming
    if (
        server_lock is not None
        and server_lock.get("client_id") != client_id
        and server_lock.get("member_object")
        and server_lock.get("member_object").transcoder is not None
    ):
        # TODO: Send an event to the client to let them know their streaming session has been killed
        if member.transcoder:
            member.transcoder.close = True
            member.transcoder = None
            # Update both room state and server state given that the number of active transcoders has changed
            if room:
                await emit_room_state_update(room)
            await emit_server_state_update()
        logger.warn(
            f"[incoming_audio] Server is locked to client {server_lock.get('client_id')}. Ignoring incoming audio from client {client_id}."
        )
        return

    if member is None or room is None:
        logger.error(
            f"[incoming_audio] Received incoming_audio from {member}, but member or room is None. This should not happen."
        )
        return

    # NOTE: bytes and bytearray are very similar, but bytes is immutable, and is what is returned by socketio
    if not isinstance(blob, bytes):
        logger.error(
            f"[incoming_audio] Received audio from {member}, but it was not of type `bytes`. type(blob) = {type(blob)}"
        )
        return

    if member.transcoder is None:
        logger.error(
            f"[incoming_audio] Received audio from {member}, but no transcoder configured to process it (member.transcoder is None). This should not happen."
        )
        return

    member.transcoder.process_incoming_bytes(
        blob, dynamic_config=member.transcoder_dynamic_config
    )

    # Send back any available model output
    # NOTE: In theory it would make sense remove this from the incoming_audio handler and
    # handle this in a dedicated thread that checks for output and sends it right away,
    # but in practice for our limited demo use cases this approach didn't add noticeable
    # latency, so we're keeping it simple for now.
    events = get_transcoder_output_events(member.transcoder)
    logger.debug(f"[incoming_audio] transcoder output events: {len(events)}")

    if len(events) == 0:
        logger.debug("[incoming_audio] No transcoder output to send")
    else:
        for e in events:
            if e["event"] == "translation_speech" and member.requested_output_type in [
                "s2s",
                "s2s&t",
            ]:
                logger.debug("[incoming_audio] Sending translation_speech event")
                await sio.emit(
                    "translation_speech", e, room=f"{room.room_id}_listeners"
                )
            elif e["event"] == "translation_text" and member.requested_output_type in [
                "s2t",
                "s2s&t",
            ]:
                logger.debug("[incoming_audio] Sending translation_text event")
                await sio.emit("translation_text", e, room=f"{room.room_id}_listeners")
            else:
                logger.error(f"[incoming_audio] Unexpected event type: {e['event']}")

    return


@sio.event
@catch_and_log_exceptions_for_sio_event_handlers
async def stop_stream(sid):
    session_data = await get_session_data(sid)
    client_id, member, room = itemgetter("client_id", "member_object", "room_object")(
        session_data
    )

    logger.debug(f"[event: stop_stream][{room}] Attempting to stop stream for {member}")

    if member is None or room is None:
        message = f"Received stop_stream from {member}, but member or room is None. This should not happen."
        logger.error(message)
        return {"status": "error", "message": message}

    # In order to stop the stream and end the transcoder thread, set close to True and unset it for the member
    if member.transcoder:
        member.transcoder.close = True
        member.transcoder = None
    else:
        message = f"Received stop_stream from {member}, but member.transcoder is None. This should not happen."
        logger.warn(message)

    # We need to emit a room state update here since room state now includes # of active transcoders
    await emit_room_state_update(room)
    # Emit a server state update now that we've changed the number of active transcoders
    await emit_server_state_update()

    return {"status": "ok", "message": "Stream stopped"}


@sio.on("clear_transcript_for_all")
@catch_and_log_exceptions_for_sio_event_handlers
async def clear_transcript_for_all(sid):
    session_data = await get_session_data(sid)

    room = session_data.get("room_object")

    if room:
        await sio.emit("clear_transcript", room=f"{room.room_id}")
    else:
        logger.error("[clear_transcript] room is None. This should not happen.")


@sio.event
@catch_and_log_exceptions_for_sio_event_handlers
async def set_name(sid, name):
    logger.info(f"[Event: set_name] name={name}")
    await sio.save_session(sid, {"name": name})
