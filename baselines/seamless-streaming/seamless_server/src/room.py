# import json
import uuid


class Room:
    def __init__(self, room_id) -> None:
        self.room_id = room_id
        # members is a dict from client_id to Member
        self.members = {}

        # listeners and speakers are lists of client_id's
        self.listeners = []
        self.speakers = []

    def __str__(self) -> str:
        return f"Room {self.room_id} ({len(self.members)} member{'s' if len(self.members) == 1 else ''})"

    def to_json(self):
        varsResult = vars(self)
        # Remember: result is just a shallow copy, so result.members === self.members
        # Because of that, we need to jsonify self.members without writing over result.members,
        # which we do here via dictionary unpacking (the ** operator)
        result = {
            **varsResult,
            "members": {key: value.to_json() for (key, value) in self.members.items()},
            "activeTranscoders": self.get_active_transcoders(),
        }

        return result

    def get_active_connections(self):
        return len(
            [m for m in self.members.values() if m.connection_status == "connected"]
        )

    def get_active_transcoders(self):
        return len([m for m in self.members.values() if m.transcoder is not None])

    def get_room_status_dict(self):
        return {
            "activeConnections": self.get_active_connections(),
            "activeTranscoders": self.get_active_transcoders(),
        }


class Member:
    def __init__(self, client_id, session_id, name) -> None:
        self.client_id = client_id
        self.session_id = session_id
        self.name = name
        self.connection_status = "connected"
        self.transcoder = None
        self.requested_output_type = None
        self.transcoder_dynamic_config = None

    def __str__(self) -> str:
        return f"{self.name} (id: {self.client_id[:4]}...) ({self.connection_status})"

    def to_json(self):
        self_vars = vars(self)
        return {
            **self_vars,
            "transcoder": self.transcoder is not None,
        }
