import sys
import os
sys.path.insert(0, os.path.abspath('../ACPC'))
import protocol_to_node

protocol_to_node = ACPCProtocolToNode()
state = protocol_to_node.parse_state("MATCHSTATE:0:99:cc/r8146:Kh|/As")
debug = 0