import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Tiep_Library_v7 import a_LIST_of__Beamforming_vectors
from Tiep_Library_v7 import a_LIST_of__AN_matrices
from Tiep_Library_v7 import get_random_string
from Tiep_Library_v7 import np__hET_b, np__hET_A_z
from Tiep_Library_v7 import np__squaredNorm_hET_A_z
from Tiep_Library_v7 import np__squaredNorm_hET_A
from Tiep_Library_v7 import np__squaredNorm_nB, np_Norm_hB
from class_neural_network_SEP_v3 import NeuralNetwork_softplus


def QAM_4(M, i):
    scaling_factor = np.sqrt(2*(M-1)/3)
    # http://www.dsplog.com/2007/09/23/scaling-factor-in-qam/
    qam = [-1-1j, -1+1j, 1+1j, 1-1j]/scaling_factor
    return np.array([qam[i]])


epsilon = 0.7  # Priority factor
n_antennas = 3
snrdB = 5
snr = 10**(snrdB/10)
M = 4
constellations = np.array([QAM_4(M, i) for i in range(M)])

n_Iterations = 1
size_batch = 64
size_out = 1
learning_rate = 0.001
n_hidden = 20
n_epochs = 201
n_realizations = 500

''' Declare objects '''
random_name = get_random_string(8)
object_NN_softplus = NeuralNetwork_softplus(random_name,
                                            epsilon,
                                            n_antennas,
                                            snrdB,
                                            size_batch,
                                            size_out,
                                            learning_rate,
                                            n_hidden)

''' Prepare containters to save values '''
# At each iteration, we want to save some corresponding values
CB_array = np.zeros([1, n_epochs])
CE_array = np.zeros([1, n_epochs])
CSEP_B_array = np.zeros([1, n_epochs])
CSEP_E_array = np.zeros([1, n_epochs])
Cs_array = np.zeros([1, n_epochs])
p_array = np.zeros([1, n_epochs])
cost_array = np.zeros([1, n_epochs])
constraint_array = np.zeros([1, n_epochs])  # [c]+ = max(0,c) where c<=0 is the constraint

# After all iterations are finished, we want to save the cumulative values
cum_CB = np.zeros([1, n_epochs])  # Cumulative value of C_B
cum_CE = np.zeros([1, n_epochs])  # Cumulative value of C_E
cum_CSEP_B = np.zeros([1, n_epochs])
cum_CSEP_E = np.zeros([1, n_epochs])
cum_cost = np.zeros([1, n_epochs])
cum_p = np.zeros([1, n_epochs])  # Cumulative value of power
cum_constraint = np.zeros([1, n_epochs])  # Cumulative value of [c]+ = max(0, c)


t0 = time.time()
''' Run Session '''
with tf.Session() as sess:
    for iteration in range(n_Iterations):
        sess.run(tf.global_variables_initializer())
        if iteration % 10 == 0:
            print("Iteration = ", iteration)
        # hB and hE in row
        hB_row_val = (np.random.normal(0, 1, [size_batch, n_antennas])
                      + 1j*np.random.normal(0, 1, [size_batch, n_antennas])
                      )/np.sqrt(2)
        hE_row_val = (np.random.normal(0, 1, [size_batch, n_antennas])
                      + 1j*np.random.normal(0, 1, [size_batch, n_antennas])
                      )/np.sqrt(2)
        nB_val = (np.random.normal(0, 1, [size_batch, 1])
                  + 1j*np.random.normal(0, 1, [size_batch, 1])
                  )/np.sqrt(2)
        z_row_val = (np.random.normal(0, 1, [size_batch, n_antennas-1])
                     + 1j*np.random.normal(0, 1, [size_batch, n_antennas-1])
                     )/np.sqrt(2)
        # h_val = np.concatenate((hB_row_val,hE_row_val), axis=1)

        # beamforming b and artificial noise matrix A
        LIST_beam_col_val = a_LIST_of__Beamforming_vectors(hB_row_val)
        LIST_AN_val = a_LIST_of__AN_matrices(hB_row_val)

        # some related values
        hET_b_val = np__hET_b(hE_row_val,
                              LIST_beam_col_val,
                              size_batch)
        hET_A_z_val = np__hET_A_z(hE_row_val,
                                  LIST_AN_val,
                                  z_row_val,
                                  size_batch)
        squaredNorm_hET_A_val = np__squaredNorm_hET_A(hE_row_val,
                                                      LIST_AN_val,
                                                      size_batch)
        squaredNorm_hET_A_z_val = np__squaredNorm_hET_A_z(hET_A_z_val,
                                                          size_batch)
        squaredNorm_nB_val = np__squaredNorm_nB(nB_val,
                                                size_batch)
        Norm_hB_val = np_Norm_hB(hB_row_val, size_batch)

        ''' TRAINING '''
        for epoch in range(n_epochs):
            object_NN_softplus.train(hB_row_val,
                                     hE_row_val,
                                     nB_val,
                                     z_row_val,
                                     LIST_beam_col_val,
                                     LIST_AN_val,
                                     hET_b_val,
                                     hET_A_z_val,
                                     squaredNorm_hET_A_val,
                                     squaredNorm_hET_A_z_val,
                                     squaredNorm_nB_val,
                                     Norm_hB_val,
                                     sess)  # Training
            BATCH_p_val = object_NN_softplus.get_output(hB_row_val,
                                                        hE_row_val,
                                                        nB_val,
                                                        z_row_val,
                                                        LIST_beam_col_val,
                                                        LIST_AN_val,
                                                        hET_b_val,
                                                        hET_A_z_val,
                                                        squaredNorm_hET_A_val,
                                                        squaredNorm_hET_A_z_val,
                                                        squaredNorm_nB_val,
                                                        Norm_hB_val,
                                                        sess)  # Power
            BATCH_cost_val = object_NN_softplus.get_cost(hB_row_val,
                                                         hE_row_val,
                                                         nB_val,
                                                         z_row_val,
                                                         LIST_beam_col_val,
                                                         LIST_AN_val,
                                                         hET_b_val,
                                                         hET_A_z_val,
                                                         squaredNorm_hET_A_val,
                                                         squaredNorm_hET_A_z_val,
                                                         squaredNorm_nB_val,
                                                         Norm_hB_val,
                                                         sess)
            BATCH_CB_val = object_NN_softplus.get_CB(hB_row_val,
                                                     hE_row_val,
                                                     nB_val,
                                                     z_row_val,
                                                     LIST_beam_col_val,
                                                     LIST_AN_val,
                                                     hET_b_val,
                                                     hET_A_z_val,
                                                     squaredNorm_hET_A_val,
                                                     squaredNorm_hET_A_z_val,
                                                     squaredNorm_nB_val,
                                                     Norm_hB_val,
                                                     sess)
            BATCH_CE_val = object_NN_softplus.get_CE(hB_row_val,
                                                     hE_row_val,
                                                     nB_val,
                                                     z_row_val,
                                                     LIST_beam_col_val,
                                                     LIST_AN_val,
                                                     hET_b_val,
                                                     hET_A_z_val,
                                                     squaredNorm_hET_A_val,
                                                     squaredNorm_hET_A_z_val,
                                                     squaredNorm_nB_val,
                                                     Norm_hB_val,
                                                     sess)
            BATCH_CSEP_B_val = object_NN_softplus.get_CSEP_B(hB_row_val,
                                                             hE_row_val,
                                                             nB_val,
                                                             z_row_val,
                                                             LIST_beam_col_val,
                                                             LIST_AN_val,
                                                             hET_b_val,
                                                             hET_A_z_val,
                                                             squaredNorm_hET_A_val,
                                                             squaredNorm_hET_A_z_val,
                                                             squaredNorm_nB_val,
                                                             Norm_hB_val,
                                                             sess)
            BATCH_CSEP_E_val = object_NN_softplus.get_CSEP_E(hB_row_val,
                                                             hE_row_val,
                                                             nB_val,
                                                             z_row_val,
                                                             LIST_beam_col_val,
                                                             LIST_AN_val,
                                                             hET_b_val,
                                                             hET_A_z_val,
                                                             squaredNorm_hET_A_val,
                                                             squaredNorm_hET_A_z_val,
                                                             squaredNorm_nB_val,
                                                             Norm_hB_val,
                                                             sess)
            # Calculate the average values over a batch of multiple examples
            p_array[0][epoch] = np.mean(BATCH_p_val)
            cost_array[0][epoch] = np.mean(BATCH_cost_val)
            CB_array[0][epoch] = np.mean(BATCH_CB_val)
            CE_array[0][epoch] = np.mean(BATCH_CE_val)
            CSEP_B_array[0][epoch] = np.mean(BATCH_CSEP_B_val)
            CSEP_E_array[0][epoch] = np.mean(BATCH_CSEP_E_val)

        ''' End of training '''
        cum_p = cum_p + p_array
        cum_cost = cum_cost + cost_array
        cum_CB = cum_CB + CB_array
        cum_CE = cum_CE + CE_array
        cum_CSEP_B = cum_CSEP_B + CSEP_B_array
        cum_CSEP_E = cum_CSEP_E + CSEP_E_array

    ''' End of iterations '''
    avg_p_array = cum_p/n_Iterations
    avg_cost_array = cum_cost/n_Iterations
    avg_CB_array = cum_CB/n_Iterations
    avg_CE_array = cum_CE/n_Iterations
    avg_Cs_array = avg_CB_array-avg_CE_array
    avg_CSEP_B_array = cum_CSEP_B/n_Iterations
    avg_CSEP_E_array = cum_CSEP_E/n_Iterations

    ''' =============== End of Simulation ==============='''
print("CB/CE = ", avg_CB_array[0][n_epochs-1]/avg_CE_array[0][n_epochs-1])
print("SEP_B = ", avg_CSEP_B_array[0][n_epochs-1])
print("SEP_E = ", avg_CSEP_E_array[0][n_epochs-1])

t1 = time.time()
runtime = t1 - t0
print('Run time: {:.2f} seconds'.format(round(runtime, 2)))


''' Show results '''
iteration = [i for i in range(len(avg_p_array[0]))]

fig1 = plt.figure(1)
plt.plot(iteration, avg_cost_array[0],
         label="tanh-based method", color='k', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average cost', fontsize=15)
plt.xlim((0, n_epochs))
plt.legend(loc='upper right', fontsize=12)
fig1.savefig('figs/Cost_vs_epoch.png', dpi=150)

fig2 = plt.figure(2)
plt.plot(iteration, avg_CB_array[0],
         label="$C_B$", color='b', linewidth=2)
plt.plot(iteration, avg_CE_array[0],
         label="$C_E$", color='r', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average capacities', fontsize=15)
plt.xlim((0, n_epochs))
plt.ylim(ymin=0, ymax=np.log2(M))
plt.legend(loc='lower right', fontsize=12)
fig2.savefig('figs/Cap_vs_epoch.png', dpi=150)

fig3 = plt.figure(3)
plt.plot(iteration, avg_CSEP_B_array[0],
         label="$P_B$", color='b', linewidth=2)
plt.plot(iteration, avg_CSEP_E_array[0],
         label="$P_E$", color='r', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average SEPs', fontsize=15)
plt.xlim((0, n_epochs))
plt.ylim(ymax=1)
plt.yscale('log')
plt.legend(loc='best', fontsize=12)
fig3.savefig('figs/SER_vs_epoch.png', dpi=150)
