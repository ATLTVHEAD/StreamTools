# mqttHotkey.py
# Recieves a hotkey string over mqtt and triggers presses the hotkey on the snapcam app  
# Written by: Nate Damen
# Created on April 9th, 2021
# Updated on April 10th, 2021

import paho.mqtt.client as mqtt
import keyboard


# The callback for when the client receives a CONNACK response from the server.
# Subscribe to the Heat clicks and the hotkey presses
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("viewer_click")
    client.subscribe("pc2/hotkeypress")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if (msg.topic == "pc2/hotkeypress"):
        #if the message is hotkey, press the hotkey sent
        keypress = msg.payload
        print(keypress.decode('UTF-8'))
        keyboard.press_and_release(keypress.decode('UTF-8'))
        #send confirmation 
        mqttc.publish("pc2/hotkeypressed", keypress.decode('UTF-8'))
    elif (msg.topic == "viewer_click"):
        #Future do an effect change based on clicks or channel points
        print(msg.payload)
    else:
        #what even is this message? 
        print(msg.topic + " " + str(msg.payload))

if __name__ == "__main__":
    #set client connect and on message functions
    mqttc = mqtt.Client()
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    #Connect to local MQTT server and loop forever
    mqttc.connect("192.168.1.106", 1883, 60)
    mqttc.loop_forever()
        
