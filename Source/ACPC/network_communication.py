'''
--- Handles network communication for DeepStack.
-- 
-- Requires [luasocket](http://w3.impa.br/~diego/software/luasocket/)
-- (can be installed with `luarocks install luasocket`).
-- @classmod network_communication
'''

# local arguments = require "Settings.arguments"
# local socket = require "socket"
import socket
import arguments
from constants import constants


#
# local ACPCNetworkCommunication = torch.class('ACPCNetworkCommunication')

class ACPCNetworkCommunication(object):
    # --- Constructor
    # function ACPCNetworkCommunication:_init()
    # end
    def __init__(self):
        print('ACPCNetworkCommunication initialized')

    '''
    --- Connects over a network socket.
    -- 
    -- @param server the server that sends states to DeepStack, and to which
    -- DeepStack sends actions
    -- @param port the port to connect on
    
    '''

    def connect(self, server, port):
        server = server or arguments.acpc_server
        port = port or arguments.acpc_server_port

        self.connection = ''
        assert (socket.connect(server, port))
        self._handshake()

    # end

    # --- Sends a handshake message to initialize network communication.
    # -- @local
    def _handshake(self):
        self.send_line("VERSION:2.0.0")

    # end

    # --- Sends a message to the server.
    # -- @param line a string to send to the server
    def send_line(self, line):
        self.connection.send(line + '\r\n')

    # end

    # --- Waits for a text message from the server. Blocks until a message is
    # -- received.
    # -- @return the message received
    def get_line(self):
        out, status = self.connection.receive('*l')

    assert (status ~= "closed")
    return out

    # end

    # --- Ends the network communication.
    def close(self):
        self.connection.close()
    # end
