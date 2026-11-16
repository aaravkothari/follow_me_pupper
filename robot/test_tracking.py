#!/usr/bin/env python3
"""
Interactive Tracking Test Script
Press 0 to disable tracking, 1 to enable tracking with object selection
"""
import rclpy
import sys
import os
import time
import threading

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from karel import KarelPupper

# Common trackable objects for quick reference
COMMON_OBJECTS = [
    "red_cap", "person", "dog", "cat", "car", "bicycle", "motorcycle",
    "bottle", "cup", "chair", "couch", "bed", "laptop",
    "cell phone", "book", "clock", "scissors", "teddy bear",
    "backpack", "umbrella", "handbag", "tie", "suitcase"
]

class InteractiveTrackingTester:
    def __init__(self):
        print("=" * 70)
        print("🎯 Interactive Tracking Test")
        print("=" * 70)
        print()
        
        # Initialize
        print("Initializing Karel Pupper...")
        if not rclpy.ok():
            rclpy.init()
        self.pupper = KarelPupper()
        print("✅ Karel initialized!")
        print()
        
        self.tracking_enabled = False
        self.current_object = None
        self.running = True
    
    def show_status(self):
        """Display current tracking status"""
        print("\n" + "─" * 70)
        if self.tracking_enabled:
            print(f"📍 Status: TRACKING - {self.current_object.upper()}")
            print("   Robot is searching for / following the target")
        else:
            print("📍 Status: IDLE")
            print("   Robot is standing still, not tracking")
        print("─" * 70)
    
    def show_menu(self):
        """Display menu options"""
        print("\nCommands:")
        print("  [0] - Disable tracking (IDLE mode)")
        print("  [1] - Enable tracking (choose object)")
        print("  [Ctrl+C] - Quit and cleanup")
        print()
    
    def show_objects(self):
        """Show list of common trackable objects"""
        print("\n📋 Common Trackable Objects:")
        print("─" * 70)
        for i, obj in enumerate(COMMON_OBJECTS, 1):
            print(f"  {i:2d}. {obj:20s}", end="")
            if i % 3 == 0:
                print()  # New line every 3 items
        print()
        print("\n   + 77 more objects! See coco.txt for full list")
        print("─" * 70)
    
    def enable_tracking(self):
        """Enable tracking with object selection"""
        self.show_objects()
        print("\nEnter object to track (e.g., 'person', 'dog', 'cat'):")
        print("Or enter a number from the list above:")
        
        try:
            choice = input("➤ ").strip()
            
            # Check if it's a number (selecting from list)
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(COMMON_OBJECTS):
                    object_name = COMMON_OBJECTS[idx]
                else:
                    print(f"❌ Invalid number. Please choose 1-{len(COMMON_OBJECTS)}")
                    return
            else:
                # Direct object name entry
                object_name = choice.lower()
            
            if not object_name:
                print("❌ No object entered")
                return
            
            print(f"\n🎯 Starting tracking: {object_name}")
            self.pupper.begin_tracking(object_name)
            # Spin once to ensure message is published
            rclpy.spin_once(self.pupper.node, timeout_sec=0.1)
            
            self.tracking_enabled = True
            self.current_object = object_name
            
            print(f"✅ Tracking enabled for '{object_name}'")
            print("   The robot will now search for and track this object!")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def disable_tracking(self):
        """Disable tracking"""
        if not self.tracking_enabled:
            print("ℹ️  Tracking is already disabled")
            return
        
        print("\n🛑 Stopping tracking...")
        self.pupper.end_tracking()
        # Spin once to ensure message is published
        rclpy.spin_once(self.pupper.node, timeout_sec=0.1)
        
        self.tracking_enabled = False
        self.current_object = None
        
        print("✅ Tracking disabled")
        print("   Robot will return to IDLE state")
        time.sleep(0.5)
    
    def run(self):
        """Main interactive loop"""
        print("\n" + "=" * 70)
        print("🎮 Interactive Control Active")
        print("=" * 70)
        print("\n💡 This script is launched by run_tracking.sh")
        print("   All required components should already be running!")
        print()
        print("Press Ctrl+C anytime to exit and cleanup automatically")
        print()
        
        try:
            while self.running:
                self.show_status()
                self.show_menu()
                
                try:
                    choice = input("Enter command: ").strip().lower()
                    
                    if choice == '0':
                        self.disable_tracking()
                    
                    elif choice == '1':
                        self.enable_tracking()
                    
                    elif choice == 'help' or choice == 'h' or choice == '?':
                        self.show_objects()
                    
                    elif choice == '':
                        # Just pressed Enter, ignore
                        continue
                    
                    else:
                        print(f"❌ Unknown command: '{choice}'")
                        print("   Use 0 or 1 (Ctrl+C to exit)")
                
                except KeyboardInterrupt:
                    print("\n\n👋 Shutting down...")
                    self.running = False
                    break
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            if self.tracking_enabled:
                self.pupper.end_tracking()
                rclpy.spin_once(self.pupper.node, timeout_sec=0.1)
            self.pupper.stop()
            rclpy.spin_once(self.pupper.node, timeout_sec=0.1)
            print("✅ Done!")
    
def main():
    """Main entry point"""
    try:
        tester = InteractiveTrackingTester()
        tester.run()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nGoodbye! 🐕")

if __name__ == "__main__":
    main()
