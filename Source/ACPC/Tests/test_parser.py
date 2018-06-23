require "ACPC.protocol_to_node"

protocol_to_node = ACPCProtocolToNode()
state = protocol_to_node.parse_state("MATCHSTATE:0:99:cc/r8146:Kh|/As")

debug = 0