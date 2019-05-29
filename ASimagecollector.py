# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim

import pprint
import os
import time
import random

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

counter = 0


for xpos in range(40,201,20):
    for ypos in range(-700,201,20):
        print(xpos," ",ypos)
        for zpos in range(0,-31,-5):
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(xpos, ypos, zpos), airsim.to_quaternion(0, 0, 0)), True)
            for hrot in range(0,12):
                for vrot in range(-1,2):
                    for rot in range(-1,2):
                        client.simSetCameraOrientation("0", airsim.to_quaternion(vrot*random.uniform(0,0.785), rot*random.uniform(0,0.785), hrot*random.uniform(0,0.785)))
                        counter=counter+1
                        r=random.randint(0,30000000)

                        responses = client.simGetImages([
                            airsim.ImageRequest("0", airsim.ImageType.DepthVis),
                            airsim.ImageRequest("0", airsim.ImageType.Scene)])

                        for i, response in enumerate(responses):
                            if i==0:
                                airsim.write_file(os.path.normpath('/temp/depth/imd_' + str(r) + "_" + str(counter) + '.png'),response.image_data_uint8)
                            else:
                                airsim.write_file(os.path.normpath('/temp/scene/ims_' + str(r) + "_" + str(counter) + '.png'),response.image_data_uint8)

client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)