import numpy as np
import openmc
import xml.etree.ElementTree as ET
import os
import shutil
import time
import sys
import openmc.deplete

import Define_OpenMC
import Define_Nektar
import datetime
# Department of Nuclear Science and Engineering
# Shanghai Jiao Tong University
# Wei Xiao
# bearsanxw@gmail.com
def createFolder():
    now_time = datetime.datetime.now()
    path = datetime.datetime.strftime(now_time,'%Y_%m_%d_%H_%M_%S')
    os.mkdir(path)
    return path


def preProcess_nek(cells_num_dic,parameters_dic,settings_dic,Initial_temp):
    settings_nek_dic = settings_dic['settings_nek_dic']
    # Preparation for Nektar++
    x,y,z = Define_Nektar.readNodesFromVtu(settings_nek_dic['file_name'])
    nodes_dic = {'x':x,'y':y,'z':z}
    # Initial temperature distribution in nodes
    temp_nodes_vec = Initial_temp*np.ones(len(x))
    fuel_nodes_index = Define_OpenMC.getFuelNodesIndex(nodes_dic,parameters_dic)

    return nodes_dic, temp_nodes_vec, fuel_nodes_index

def main_Depletion(cells_num_dic,parameters_dic,settings_dic,Initial_temp,controlRod_deep,empty_reflector_height):
    # Create the folder for saving data
    path = createFolder()
    # Set initial temperature matrix for OpenMC
    temp_cells_mat = Initial_temp*np.ones((cells_num_dic['n_h'],(cells_num_dic['n_r_outer']+cells_num_dic['n_r'])))
    # Initialization for OpenMC
    volume_mat,fuel_cell_ID_list = Define_OpenMC.define_Geo_Mat_Set(cells_num_dic,parameters_dic,settings_dic,temp_cells_mat,controlRod_deep,empty_reflector_height)
    # Initialization for Nektar++
    nodes_dic, temp_nodes_vec, fuel_nodes_index = preProcess_nek(cells_num_dic,parameters_dic,settings_dic,Initial_temp)
    file_name = settings_dic['settings_nek_dic']['file_name']
    solver_name = settings_dic['settings_nek_dic']['solver_name']
    temp_pipe = 1073.5

    # Maximum iteration number
    iteration = settings_dic['iteration']
    # Number of cells in OpenMC
    num_col = cells_num_dic['n_h']*(cells_num_dic['n_r']+cells_num_dic['n_r_outer'])

    # Settings for depletion calculation
    time_steps = settings_dic['time_steps']
    chain = settings_dic['chain']
    power = parameters_dic['heat_power']/8

    cal_step = 0

    for j in range(len(time_steps)+1):
        print("====================Burn-up step"+str(j+1)+" begins====================")
        file_path = path + '/'+ 'Burn-up-step-'+str(j+1)
        os.mkdir(file_path)
        # Information storage
        temp_error_vec = np.zeros(iteration)

        k_eff_mean_vec = np.zeros(iteration)
        k_eff_dev_vec = np.zeros(iteration)

        heat_mean_mat = np.zeros((iteration,num_col))
        heat_dev_mat = np.zeros((iteration,num_col))
        flux_mean_mat = np.zeros((iteration,num_col))
        flux_dev_mat = np.zeros((iteration,num_col))

        # Sub step in coupling
        sub_time_step = [0]

        for i in range(iteration):
            start_tot = time.time()

            geometry = openmc.Geometry.from_xml()
            settings = openmc.Settings.from_xml()

            if cal_step == 0:
                print('Asuka')
                operator = openmc.deplete.Operator(geometry, settings, chain)
                integrator = openmc.deplete.PredictorIntegrator(operator, sub_time_step, power)
            elif cal_step > 0 and i > 0:
                print('Momoko')
                results = openmc.deplete.ResultsList.from_hdf5("./depletion_results.h5")
                operator = openmc.deplete.Operator(geometry, settings, chain,prev_results=results)
                integrator = openmc.deplete.PredictorIntegrator(operator, sub_time_step, power)
            else:
                print('Hinako')
                results = openmc.deplete.ResultsList.from_hdf5("./depletion_results.h5")
                operator = openmc.deplete.Operator(geometry, settings, chain, prev_results=results)
                integrator = openmc.deplete.PredictorIntegrator(operator, [time_steps[j-1]], power)

            print("====================Iteration"+str(i+1)+" begins====================")
            temp_cells_mat_last = temp_cells_mat
            # Run OpenMC !
            time_start = time.time()
            integrator.integrate()
            time_end = time.time()
            print('Iteration '+str(i+1)+': OpenMC run time:'+str(time_end-time_start))

            results = openmc.deplete.ResultsList.from_hdf5("./depletion_results.h5")
            time_line, k = results.get_eigenvalue()
            k_eff = k[cal_step][0]
            k_eff_dev = k[cal_step][1]
            print('Iteration '+str(i+1)+': k_eff:'+str(k_eff)+ ' k_eff_dev:'+str(k_eff_dev))

            # Post-process the result of heat source and generate Force.pts
            time_start = time.time()
            k_eff_comb,tally_dic = Define_OpenMC.postProcess(nodes_dic,volume_mat,temp_nodes_vec,fuel_nodes_index,parameters_dic,cells_num_dic,settings_dic,fuel_cell_ID_list,cal_step)
            time_end = time.time()
            print('Iteration '+str(i+1)+': OpenMC post-process time:'+str(time_end-time_start))

            # k-eff: mean value and standard deviation
            k_eff_mean_vec[i] = k_eff
            k_eff_dev_vec[i] = k_eff_dev
          
            # Heat and flux    
            heat_mean_mat[i,:] = tally_dic['heat_mean']
            heat_dev_mat[i,:] = tally_dic['heat_dev']
            flux_mean_mat[i,:] = tally_dic['flux_mean']
            flux_dev_mat[i,:] = tally_dic['flux_dev']

            # Run Nektar++
            if os.path.exists(file_name+'.fld'):
                os.remove(file_name+'.fld')
            if os.path.exists(file_name+'.vtu'):
                os.remove(file_name+'.vtu')

            time_start = time.time()
            file_name_new = Define_Nektar.runNektar_Temp(file_name,solver_name,temp_pipe,i)
            time_end = time.time()
            print('Iteration '+str(i+1)+' Nektar run time:'+str(time_end-time_start))

            # Post-process result of temperature
            time_start = time.time()
            temp_nodes_vec = Define_Nektar.postProcess_Temp(file_name_new)
            time_end = time.time()
            print('Iteration '+str(i+1)+' Nektar post-process time:'+str(time_end-time_start))
            temp_cells_mat = Define_OpenMC.getCellTemperature(nodes_dic,temp_nodes_vec,fuel_nodes_index,parameters_dic,cells_num_dic)

            volume_mat,fuel_cell_ID_list = Define_OpenMC.define_Geo_Mat_Set(cells_num_dic,parameters_dic,settings_dic,temp_cells_mat,controlRod_deep,empty_reflector_height)

            # Largest relative error of fuel temperature
            temp_error = np.abs(temp_cells_mat-temp_cells_mat_last)/temp_cells_mat_last
            temp_error_vec[i] = temp_error.max()
            print('Maximum relative error: '+str(temp_error_vec[i]))

            end_tot = time.time()
            print('Total time: '+str(end_tot-start_tot))

            # Move the result of OpenMC 
            if os.path.exists("openmc_simulation_n"+str(cal_step)+".h5"):
                shutil.copy(("openmc_simulation_n"+str(cal_step)+".h5"),file_path)

            if cal_step == 0:
                cal_step = cal_step+2
            else:
                cal_step = cal_step+1

            if temp_error.max()<1e-2:
                break


        # Save data

        np.savetxt((file_path+'/'+'temp_error_vec.txt'),temp_error_vec)

        np.savetxt((file_path+'/'+'k_eff_mean_vec.txt'),k_eff_mean_vec)
        np.savetxt((file_path+'/'+'k_eff_dev_vec.txt'),k_eff_dev_vec)

        np.savetxt((file_path+'/'+'heat_mean_mat.txt'),heat_mean_mat)
        np.savetxt((file_path+'/'+'heat_dev_mat.txt'),heat_dev_mat)
        np.savetxt((file_path+'/'+'flux_mean_mat.txt'),flux_mean_mat)
        np.savetxt((file_path+'/'+'flux_dev_mat.txt'),flux_dev_mat)

        # Move results of Nektar++
        if os.path.exists(file_name+'.fld'):
            shutil.copy((file_name+'.fld'),file_path)
        if os.path.exists(file_name+'.vtu'):
            shutil.copy((file_name+'.vtu'),file_path)


