'''''
--- Handles communication to and from DeepStack using the ACPC protocol.
--
-- For details on the ACPC protocol, see
-- <http://www.computerpokercompetition.org/downloads/documents/protocols/protocol.pdf>.
-- @classmod acpc_game
'''''

# require 'ACPC.network_communication'
# require 'ACPC.protocol_to_node'
#
# local arguments = require 'Settings.arguments'
# local constants = require "Settings.constants"
import sys

sys.path.insert(0, '../Settings')
sys.path.insert(0, '../Game')
sys.path.insert(0, '../ACPC')
import arguments
import constants

import network_communication
import protocol_to_node

# local arguments = require 'Settings.arguments'
# local constants = require "Settings.constants"


# --if you want to fake what messages the acpc dealer sends, put them in the following list and uncomment it.
debug_msg = None #- -{"MATCHSTATE:0:99::Kh|/", "MATCHSTATE:0:99:cr200:Kh |/", "MATCHSTATE:0:99:cr200:Kh|/Ks"}


class ACPCGame(object):

    # --- Constructor
    # function ACPCGame:__init()
    # end
    def __init__(self):
        print('constructor of ACPCGame')

    # --- Connects to a specified ACPC server which acts as the dealer.
    # --
    # -- @param server the server that sends states to DeepStack, which responds
    # -- with actions
    # -- @param port the port to connect on
    # -- @see network_communication.connect
    def connect(self, server, port):
        self.protocol_to_node = ACPCProtocolToNode()
        if not debug_msg:
            self.network_communication = ACPCNetworkCommunication()
            self.network_communication.connect(server, port)

    #   end
    # end

    # --- Receives and parses the next poker situation where DeepStack must act.
    # --
    # -- Blocks until the server sends a situation where DeepStack acts.
    # -- @return the parsed state representation of the poker situation (see
    # -- @{protocol_to_node.parse_state})
    # -- @return a public tree node for the state (see
    # -- @{protocol_to_node.parsed_state_to_node})
    def get_next_situation(self):
        while True:
            msg = None

            # --1.0 get the message from the dealer
            if not debug_msg:
                msg = self.network_communication.get_line()
            else:
                msg = table.remove(debug_msg, 1)
            # end

            print("Received acpc dealer message:")
            print(msg)

            # --2.0 parse the string to our state representation
            parsed_state = self.protocol_to_node.parse_state(msg)

            # --3.0 figure out if we should act

            # --current player to act is us
            if parsed_state.acting_player == parsed_state.position:
                # --we should not act since this is an allin situations
                if parsed_state.bet1 == parsed_state.bet2 and parsed_state.bet1 == arguments.stack:
                    print("Not our turn - alling")
                # --we should act
                else:
                    print("Our turn")

                    self.last_msg = msg
                    # --create a tree node from the current state
                    node = self.protocol_to_node.parsed_state_to_node(parsed_state)

                    return parsed_state, node
                # end
            # --current player to act is the opponent
            else:
                print("Not our turn")
            # end
        # end

    # end

    # --- Informs the server that DeepStack is playing a specified action.
    # -- @param adviced_action a table specifying the action chosen by Deepstack,
    # -- with the fields:
    # --
    # -- * `action`: an element of @{constants.acpc_actions}
    # --
    # -- * `raise_amount`: the number of chips raised (if `action` is raise)
    def play_action(self, adviced_action):
        message = self.protocol_to_node.action_to_message(self.last_msg, adviced_action)
        print("Sending a message to the acpc dealer:")
        print(message)

        if not debug_msg:
            self.network_communication.send_line(message)
    #   end
    # end
