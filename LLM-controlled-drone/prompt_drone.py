#!/usr/bin/env python3
"""Interactive Terminal Prompt for LLM-Controlled Drone.

Run this in any terminal to control the drone with natural language prompts:
  python3 /home/tushar/Desktop/swarm/LLM-controlled-drone/prompt_drone.py
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    rclpy.init()
    node = Node('drone_prompt_cli')
    pub = node.create_publisher(String, '/user_command', 10)

    print('\n' + '=' * 65)
    print('  AI-CONTROLLED DRONE — INTERACTIVE COMMAND TERMINAL')
    print('=' * 65)
    print('Examples you can type:')
    print('  - Take off to 15 metres')
    print('  - Fly 20 metres north')
    print('  - Fly to 25m and circle the area')
    print('  - Search for the person')
    print('  - Search for the car')
    print('  - Fly a 30 metre square pattern')
    print('  - Hover here')
    print('  - Return to home and land')
    print('Type "quit" or Ctrl+C to exit.\n')

    # If a prompt was passed as CLI arguments, send it after DDS discovery
    if len(sys.argv) > 1:
        cmd_text = ' '.join(sys.argv[1:]).strip()
        msg = String()
        msg.data = cmd_text
        import time
        for _ in range(12):
            rclpy.spin_once(node, timeout_sec=0.1)
            if pub.get_subscription_count() > 0:
                break
            time.sleep(0.1)
        for _ in range(3):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.15)
        print(f'>>> Sent prompt to /user_command: "{cmd_text}" (subscribers={pub.get_subscription_count()})')
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        while rclpy.ok():
            cmd_text = input('Drone Prompt > ').strip()
            if not cmd_text:
                continue
            if cmd_text.lower() in ('exit', 'quit', 'q'):
                break
            msg = String()
            msg.data = cmd_text
            pub.publish(msg)
            print(f'  [Sent] -> "{cmd_text}" (watch Gazebo + Live 2D Map GUI!)')
    except (KeyboardInterrupt, EOFError):
        print('\nExiting prompt CLI.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
