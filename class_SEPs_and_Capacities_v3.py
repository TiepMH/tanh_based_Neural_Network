import tensorflow as tf
import numpy as np
from Tiep_Library_v7 import Norm_of_a_complex_vector
from Tiep_Library_v7 import squaredNorm_of_a_complex_vector
# from Tiep_Library_v7 import squaredNorm_of_complex_uT_times_v
# from Tiep_Library_v7 import a_batch_of__Norm_hET_beam
# from Tiep_Library_v7 import a_batch_of__squaredNorm_hET_A
# from Tiep_Library_v7 import a_batch_of__hET_A_z, a_batch_of__hET_beam


class SEPs:

    def __init__(self, epsilon,
                 n_antennas,
                 snrdB,
                 size_batch,
                 hB_row,
                 hE_row,
                 nB,
                 z_row,
                 LIST_beam_col,
                 LIST_AN,
                 hET_b,
                 hET_A_z,
                 squaredNorm_hET_A,
                 squaredNorm_hET_A_z,
                 squaredNorm_nB,
                 Norm_hB,
                 NN_outputs):
        # System Parameters
        self.snr = 10.0**(snrdB/10.0)
        self.sigma2_B = 1.0
        self.sigma2_E = 0.001

        # Priority factor
        self.epsilon = epsilon

        # QAM parameters
        self.M = 4
        self.scaling_factor = tf.sqrt(2.0*(self.M-1.0)/3.0)
        self.M_min = 2.0
        self.d_min = 2.0/self.scaling_factor
        self.P_Emax = (self.M - 1.0)/self.M

        # Compute CSEP_Bob
        self.CSEP_B = tf.multiply(
                        self.M_min,
                        self.Q_func(
                            tf.multiply(
                                    tf.multiply(
                                        tf.sqrt(NN_outputs*self.snr), Norm_hB),
                                    self.d_min
                                )/tf.sqrt(2.0)
                            )
                        )

        # Compute CSEP_Eve
        self.nume = tf.multiply(
                        tf.multiply(tf.sqrt(NN_outputs*self.snr),
                                    Norm_of_a_complex_vector(hET_b)),
                        self.d_min
                    )

        self.deno = tf.sqrt(
                        tf.multiply(
                            2.0,
                            tf.add(
                                tf.multiply(
                                    tf.multiply(tf.subtract(1.0, NN_outputs),
                                                self.snr),
                                    squaredNorm_hET_A
                                )/(n_antennas - 1),
                                self.sigma2_E/self.sigma2_B
                            )
                        )
                    )

        self.CSEP_E = tf.multiply(self.M_min, self.Q_func(self.nume/self.deno))

        # Compute the channel capacity for Bob
        self.C_B = self.Capacity_B(n_antennas, size_batch, hB_row, hE_row,
                                   nB,
                                   squaredNorm_nB, Norm_hB,
                                   NN_outputs)

        # Compute the channel capacity for Eve
        self.C_E = self.Capacity_E(n_antennas,
                                   size_batch,
                                   hB_row,
                                   hE_row,
                                   z_row,
                                   LIST_beam_col,
                                   LIST_AN,
                                   hET_b,
                                   hET_A_z,
                                   squaredNorm_hET_A,
                                   squaredNorm_hET_A_z,
                                   NN_outputs)

        # The constraint is C_B - C_E >= 0
        self.Cs = tf.subtract(self.C_B, self.C_E)
        self.c_pos = tf.maximum(self.Cs, 0.0)

        # Objective function
        self.objective_function = tf.add(
                                    tf.multiply(self.epsilon, self.CSEP_B),
                                    tf.multiply(
                                                1.0 - self.epsilon,
                                                tf.abs(tf.subtract(self.P_Emax,
                                                                   self.CSEP_E))
                                                )
                                    )

    # ================================================================
    def Psi_B(self, s_i, s_j,
              n_antennas,
              size_batch,
              hB_row,
              hE_row,
              nB,
              squaredNorm_nB,
              Norm_hB,
              NN_outputs):
        term_1 = squaredNorm_of_a_complex_vector(
                    tf.add(
                        tf.multiply(
                            tf.cast(tf.multiply(
                                tf.sqrt(tf.multiply(NN_outputs, self.snr)),
                                tf.cast(Norm_hB, dtype=tf.float32)
                                ), dtype=tf.complex64),
                            s_i - s_j
                            ),
                        nB
                    )
                )
        Psi_ij = tf.subtract(tf.cast(squaredNorm_nB, dtype=tf.float32),
                             term_1)
        return Psi_ij

    # ================================================================
    def Capacity_B(self, n_antennas,
                   size_batch,
                   hB_row,
                   hE_row,
                   nB,
                   squaredNorm_nB, Norm_hB,
                   NN_outputs):
        # M = 4
        scaling_factor = np.sqrt(2*(self.M-1)/3)
        qam_4 = [-1-1j, -1+1j, 1+1j, 1-1j]/scaling_factor
        #
        sum_of__log = 0.0
        for s_i in qam_4:
            sum_of__e_power_Psi = 0.0
            for s_j in qam_4:
                sum_of__e_power_Psi += tf.exp(self.Psi_B(s_i, s_j,
                                                         n_antennas,
                                                         size_batch,
                                                         hB_row, hE_row,
                                                         nB,
                                                         squaredNorm_nB,
                                                         Norm_hB,
                                                         NN_outputs))
            sum_of__log += tf.cast(tf.log(sum_of__e_power_Psi),
                                   dtype=tf.float32)/tf.log(2.0)
        kq = tf.subtract(
                    tf.log(tf.cast(self.M, dtype=tf.float32))/tf.log(2.0),
                    tf.multiply(1/self.M, sum_of__log)
                )
        return kq

    # ================================================================
    def Psi_E(self, s_i, s_j,
              n_antennas,
              size_batch,
              hB_row,
              hE_row,
              z_row,
              LIST_beam_col,
              LIST_AN,
              hET_b,
              hET_A_z,
              squaredNorm_hET_A,
              squaredNorm_hET_A_z,
              NN_outputs):
        term_1 = tf.add(
                        tf.multiply(
                            tf.multiply(
                                tf.cast(tf.sqrt(
                                    tf.multiply(NN_outputs, n_antennas-1)
                                    / (1-NN_outputs)),
                                    dtype=tf.complex64),
                                tf.cast(hET_b, dtype=tf.complex64)
                                ),
                            s_i - s_j
                        ),
                        hET_A_z
                    )
        squaredNorm_term1 = squaredNorm_of_a_complex_vector(term_1)
        Psi_ij = tf.subtract(squaredNorm_hET_A_z, squaredNorm_term1)/squaredNorm_hET_A
        return Psi_ij

    # ================================================================
    def Capacity_E(self, n_antennas,
                   size_batch,
                   hB_row,
                   hE_row,
                   z_row,
                   LIST_beam_col,
                   LIST_AN,
                   hET_b,
                   hET_A_z,
                   squaredNorm_hET_A,
                   squaredNorm_hET_A_z,
                   NN_outputs):
        #
        scaling_factor = np.sqrt(2*(self.M-1)/3)
        QAM = [-1-1j, -1+1j, 1+1j, 1-1j]/scaling_factor
        #
        sum_of__log = 0.0
        for s_i in QAM:
            sum_of__e_power_Psi = 0.0
            for s_j in QAM:
                sum_of__e_power_Psi += tf.exp(self.Psi_E(s_i, s_j,
                                                         n_antennas,
                                                         size_batch,
                                                         hB_row,
                                                         hE_row,
                                                         z_row,
                                                         LIST_beam_col,
                                                         LIST_AN,
                                                         hET_b,
                                                         hET_A_z,
                                                         squaredNorm_hET_A,
                                                         squaredNorm_hET_A_z,
                                                         NN_outputs)
                                              )
            sum_of__log += tf.log(sum_of__e_power_Psi)/tf.log(2.0)
        kq = tf.subtract(
                    tf.log(tf.cast(self.M, dtype=tf.float32))/tf.log(2.0),
                    tf.multiply(1/self.M, sum_of__log)
                )
        return kq

    # ================================================================
    def Q_func(self, z):
        return 0.5*tf.math.erfc(z/tf.sqrt(2.0))
