import tensorflow as tf
import numpy as np
from scipy.linalg import null_space
import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(length))
    return random_string 


'''Operations on complex variables '''
def col_vector(row_vector, length, conjugate=False):
    if conjugate == True:
        col = np.reshape( row_vector.conj() , [length,1] )
    else:
        col = np.reshape( row_vector , [length,1] )
    return col

def Norm_of_a_complex_vector(u_row):
    uH = tf.conj(u_row)
    product_elementwise = tf.multiply(u_row, uH)
    product_elementwise = tf.real( product_elementwise ) #tf.real( a + 0j ) = a
    squaredNorm_of_u = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    Norm_of_u = (squaredNorm_of_u)**0.5
    return Norm_of_u

def squaredNorm_of_a_complex_vector(u_row):
    uH = tf.conj(u_row)
    product_elementwise = tf.multiply(u_row, uH)
    product_elementwise = tf.real( product_elementwise ) #tf.real( a + 0j ) = a 
    squaredNorm_of_u = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    return squaredNorm_of_u

def squaredNorm_of_complex_uH_times_v(u_row, v_row):
    uH = tf.conj(u_row)
    product_elementwise = tf.multiply(uH,v_row)
    sum_of_products = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    squaredNorm = tf.real(sum_of_products)**2 + tf.imag(sum_of_products)**2
    return squaredNorm

def Norm_of_complex_uH_times_v(u_row, v_row): 
    uH = tf.conj(u_row)
    product_elementwise = tf.multiply(uH,v_row)
    sum_of_products = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    Norm_ = tf.abs(sum_of_products)
    return Norm_

def squaredNorm_of_complex_uT_times_v(u_row, v_row):
    product_elementwise = tf.multiply(u_row,v_row)
    sum_of_products = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    squaredNorm = tf.real(sum_of_products)**2 + tf.imag(sum_of_products)**2
    return squaredNorm

def Norm_of_complex_uT_times_v(u_row, v_row):
    product_elementwise = tf.multiply(u_row,v_row)
    sum_of_products = tf.reduce_sum(product_elementwise, axis=1, keepdims=True)
    Norm_ = tf.abs(sum_of_products)
    return Norm_

def uH_A_u___can_be_complex_if_A_is_not_Hermitian(u_row, A, size_batch):
    u_row = tf.cast(u_row, dtype=tf.complex64)
    A = tf.cast(A, dtype=tf.complex64) #tf.matmul requires 2 arguments have the same type
    uH_A_u_MANY_EXAMPLES = []
    for idx in range(size_batch):
        u_row_SINGLE_EXAMPLE = u_row[idx,:] 
        u_row_SINGLE_EXAMPLE = tf.reshape(u_row_SINGLE_EXAMPLE, 
                                          [1,tf.size(u_row_SINGLE_EXAMPLE)]
                                          )
                            #a = [1, 2, 3] is converted into a = [[1, 2, 3]]
        uH_SINGLE_EXAMPLE = tf.conj(u_row_SINGLE_EXAMPLE)
        uH_A_SINGLE_EXAMPLE = tf.matmul(uH_SINGLE_EXAMPLE, A)
        uH_A_u_SINGLE_EXAMPLE = tf.reduce_sum(tf.multiply(uH_A_SINGLE_EXAMPLE, u_row_SINGLE_EXAMPLE),
                                              axis=1, keepdims=True)
        uH_A_u_MANY_EXAMPLES.append(uH_A_u_SINGLE_EXAMPLE)  
        ''' end of loop'''
    uH_A_u_MANY_EXAMPLES = tf.reshape(uH_A_u_MANY_EXAMPLES, [size_batch,1])
    return uH_A_u_MANY_EXAMPLES

def uH_A_u___is_real_if_A_is_Hermitian(u,A,size_batch):
    matrix = tf.real(uH_A_u___can_be_complex_if_A_is_not_Hermitian(u,A,size_batch))
    return matrix

def remove_zero_imag_part(A):
    real_A = np.real(A)    
    if np.allclose(A, real_A, rtol=1e-05, atol=1e-07) :
        return np.real(A)
    return A

def is_symmetric(A, tolerance):
    if tolerance < 1e-7:
        tol = tolerance
    else:
        tol = 1e-7
    return tf.reduce_all( tf.abs(A-tf.transpose(A)) < tol )

def is_Hermitian(A, tolerance):
    if tolerance < 1e-7:
        tol = tolerance
    else:
        tol = 1e-7
    return tf.reduce_all( tf.abs(A-tf.transpose(A, conjugate=True)) < tol )

def squaredNorm_of_complex_xH_times_A(x_row,A):
    x_row = tf.cast(x_row, dtype=tf.complex64)
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    A = tf.cast(A, dtype=tf.complex64)
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xH_times_A = tf.matmul(tf.conj(x_row),A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    REAL_PART_of_xH_times_A = tf.real(xH_times_A)
    # REAL_PART_of_xH_times_A = [ [ real(h_1), real(h_2), ..., real(h_N) ]   example 1
    #                             [ real(h_1), real(h_2), ..., real(h_N) ]   example 2 
    #                                ....... , ........ , ..., ........  
    #                             [ real(h_1), real(h_2), ..., real(h_N) ] ] example N
    IMAGINARY_PART_of_xH_times_A = tf.imag(xH_times_A)
    # IMAGINARY_PART_of_xH_times_A = [ [ im(h_1), im(h_2), ..., im(h_N) ]   example 1
    #                                  [ im(h_1), im(h_2), ..., im(h_N) ]   example 2 
    #                                ....... , ........ , ..., ........  
    #                                  [ im(h_1), im(h_2), ..., im(h_N) ] ] example N
    squared_REAL_PART_of_xH_times_A_element_wise = REAL_PART_of_xH_times_A**2
    squared_IMAGINARY_PART_of_xH_times_A_element_wise = IMAGINARY_PART_of_xH_times_A**2
    #################################################
    sum_squared_REAL_PART = tf.reduce_sum(squared_REAL_PART_of_xH_times_A_element_wise, axis=1)
    sum_squared_IMAGINARY_PART = tf.reduce_sum(squared_IMAGINARY_PART_of_xH_times_A_element_wise, axis=1)
    #################################################
    squaredNorm = sum_squared_REAL_PART + sum_squared_IMAGINARY_PART 
    # squaredNorm = [ [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] example 1
    #                 [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] example 2
    #                    ......................................................
    #                 [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] ] example N
    return tf.reshape(squaredNorm, [tf.size(squaredNorm),1])

def Norm_of_complex_xH_times_A(x_row,A):
    Norm_ = tf.sqrt( squaredNorm_of_complex_xH_times_A(x_row,A) )
    # Norm_ = [ [ (real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ...)^0.5 ] example 1
    #           [ (real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ...)^0.5 ] example 2
    #             ......................................................
    #           [ (real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ...)^0.5 ] ] example N
    return Norm_ 

def squaredNorm_of_complex_xT_times_A(x_row,A):
    x_row = tf.cast(x_row, dtype=tf.complex64)
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    A = tf.cast(A, dtype=tf.complex64)
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xT_times_A = tf.matmul(x_row,A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    REAL_PART_of_xH_times_A = tf.real(xT_times_A)
    # REAL_PART_of_xH_times_A = [ [ real(h_1), real(h_2), ..., real(h_N) ]   example 1
    #                             [ real(h_1), real(h_2), ..., real(h_N) ]   example 2 
    #                                ....... , ........ , ..., ........  
    #                             [ real(h_1), real(h_2), ..., real(h_N) ] ] example N
    IMAGINARY_PART_of_xH_times_A = tf.imag(xT_times_A)
    # IMAGINARY_PART_of_xH_times_A = [ [ im(h_1), im(h_2), ..., im(h_N) ]   example 1
    #                                  [ im(h_1), im(h_2), ..., im(h_N) ]   example 2 
    #                                ....... , ........ , ..., ........  
    #                                  [ im(h_1), im(h_2), ..., im(h_N) ] ] example N
    squared_REAL_PART_of_xH_times_A_element_wise = REAL_PART_of_xH_times_A**2
    squared_IMAGINARY_PART_of_xH_times_A_element_wise = IMAGINARY_PART_of_xH_times_A**2
    #################################################
    sum_squared_REAL_PART = tf.reduce_sum(squared_REAL_PART_of_xH_times_A_element_wise, axis=1)
    sum_squared_IMAGINARY_PART = tf.reduce_sum(squared_IMAGINARY_PART_of_xH_times_A_element_wise, axis=1)
    #################################################
    squaredNorm = sum_squared_REAL_PART + sum_squared_IMAGINARY_PART 
    # squaredNorm = [ [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] example 1
    #                 [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] example 2
    #                    ......................................................
    #                 [ real(h_1)^2 + im(h_1)^2 + real(h_2)^2 + im(h_2)^2 + ... ] ] example N
    return tf.reshape(squaredNorm, [tf.size(squaredNorm),1])

def a_complex_orthonormal_basis_containing_a_given_vector(v_0_col):
    #We perform the Gram-Schmidt process in the complex-valued field
    #Given the COLUMN vector u_0_col = v_0_col, FIND THE REMAINING COLUMN VECTORS u_i_col
    #Then normalize u_0_col, u_1_col, ..., u_{N-1}_col into [b_0, b_1, ..., b_{N-1}]
    #Note that b_0 is the beamforming vector in our paper
    #[b_1, b_2, ..., b_{N-1}] for the Artificial Noise in our paper
    #The basis = [beamforming, artificial noise matrix]
    n = v_0_col.shape[0] #The number of elements in the COLUMN vector v0_col
    basis = [] # basis = [b_0, b_1, b_2, ..., b_{N-1}]
    for i in range(n):
        if i == 0:
            b_0 = v_0_col/np.linalg.norm(v_0_col) #the normalized BEAMFORMING vector
            basis.append(b_0)
        else:
            v_i_col = (np.random.normal(0,1, [n, 1]) + 1j*np.random.normal(0,1, [n, 1]))/np.sqrt(2) #create a random vector
            u_i_col = v_i_col - sum( np.matmul(v_i_col.T, b.conj())*b  for b in basis )
            b_i = u_i_col/np.linalg.norm(u_i_col) #a column vector in ARTIFICIAL NOISE MATRIX 
            basis.append(b_i)
    return np.array(basis)

def find_artificial_noise(beamforming_col_normalized):
    n = beamforming_col_normalized.shape[0] #The number of elements beamforming_col_normalized
    #beamforming_col_normalized must be normalized to have l2_norm = 1
    OB = a_complex_orthonormal_basis_containing_a_given_vector(beamforming_col_normalized)
    artificial_noise = OB[1:] #remove the first column K[0], the remaining columns form the AN matrix
    artificial_noise = np.reshape(artificial_noise, [n-1,n]).T
    return artificial_noise 

### The 4 following functions are for the special case when the beamforming vector ...
### ... is designed as the conjugate transpose of the normalized channel of Bob 
def a_LIST_of__Beamforming_vectors(hB_row):
    size_batch = hB_row.shape[0]
    n_antennas = hB_row.shape[1]
    beamforming_vectors = []
    for batch in range(size_batch):
        hB_row_1_EX = hB_row[batch,:]
        hB_row_1_EX = np.reshape(hB_row_1_EX, [1,len(hB_row_1_EX)])
        beamforming_col = col_vector(hB_row_1_EX, n_antennas, conjugate=True)
        beamforming_col_normalized = beamforming_col / np.linalg.norm(beamforming_col)
        beamforming_vectors.append( beamforming_col_normalized )
    return beamforming_vectors

def a_LIST_of__AN_matrices(hB_row):
    size_batch = hB_row.shape[0]
    n_antennas = hB_row.shape[1]
    AN_matrices = []
    for batch in range(size_batch):
        hB_row_1_EX = hB_row[batch,:]
        hB_row_1_EX = np.reshape(hB_row_1_EX, [1,len(hB_row_1_EX)])
        beamforming_col = col_vector(hB_row_1_EX, n_antennas, conjugate=True)
        beamforming_col_normalized = beamforming_col / np.linalg.norm(beamforming_col)
        # AN_matrix = find_artificial_noise(beamforming_col_normalized)      
        AN_matrix = null_space( beamforming_col_normalized.T.conj() )
        AN_matrices.append(AN_matrix)     
    return AN_matrices 

def a_batch_of__squaredNorm_hET_beam(hE_row, LIST_beamforming_col, size_batch):
    squaredNorm_hET_beam_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = tf.reshape( hE_row[i,:] , [1, tf.size(hE_row[i,:])] )  
        beam_col_1_Example = LIST_beamforming_col[i]
        beam_row_1_Example = tf.transpose(beam_col_1_Example)
        squaredNorm_hET_beam = squaredNorm_of_complex_uT_times_v( hE_row_1_Example, beam_row_1_Example )
        squaredNorm_hET_beam_Multiple_Examples.append(squaredNorm_hET_beam)
    return tf.reshape(squaredNorm_hET_beam_Multiple_Examples, [size_batch,1])

def a_batch_of__squaredNorm_hET_A(hE_row, A, size_batch):
    squaredNorm_hET_A_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = tf.reshape( hE_row[i,:] , [1, tf.size(hE_row[i,:])] )  
        A_1_Example = A[i,:,:]
        squaredNorm_hET_A = squaredNorm_of_complex_xT_times_A( hE_row_1_Example, A_1_Example )
        squaredNorm_hET_A_Multiple_Examples.append(squaredNorm_hET_A)
    return tf.reshape(squaredNorm_hET_A_Multiple_Examples, [size_batch,1])

def a_batch_of__Norm_hET_beam(hE_row, LIST_beamforming_col, size_batch):
    Norm_hET_beam_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = tf.reshape( hE_row[i,:] , [1, tf.size(hE_row[i,:])] )  
        beam_col_1_Example = LIST_beamforming_col[i]
        beam_row_1_Example = tf.transpose(beam_col_1_Example)
        Norm_hET_beam = Norm_of_complex_uT_times_v( hE_row_1_Example, beam_row_1_Example )
        Norm_hET_beam_Multiple_Examples.append(Norm_hET_beam)
    return tf.reshape(Norm_hET_beam_Multiple_Examples, [size_batch,1])

###
def complex_xT_times_A(x_row,A):
    x_row = tf.cast(x_row, dtype=tf.complex64)
    A = tf.cast(A, dtype=tf.complex64)
    xT_times_A = tf.matmul(x_row,A) 
    return xT_times_A

def a_batch_of__hET_A_z(hE_row, LIST_AN, z_row, size_batch):
    hET_A_z_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = tf.reshape( hE_row[i,:] , [1, tf.size(hE_row[i,:])] )  
        A_1_Example = LIST_AN[i,:,:]
        z_col_1_Example = tf.reshape( z_row[i,:], [tf.size(z_row[i,:]),1] )
        hET_A = complex_xT_times_A( hE_row_1_Example, A_1_Example )
        hET_A_z = tf.matmul(hET_A, z_col_1_Example) 
        hET_A_z_Multiple_Examples.append(hET_A_z)
    return tf.reshape(hET_A_z_Multiple_Examples, [size_batch,1])

def a_batch_of__hET_beam(hE_row, LIST_beamforming_col, size_batch):
    hET_beam_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = tf.reshape( hE_row[i,:] , [1, tf.size(hE_row[i,:])] )
        beam_col_1_Example = LIST_beamforming_col[i]
        hET_beam = tf.matmul(hE_row_1_Example, beam_col_1_Example)
        hET_beam_Multiple_Examples.append(hET_beam)
    return tf.reshape(hET_beam_Multiple_Examples, [size_batch,1])

''' Operations based on NUMPY '''
def np__hET_A_z(hE_row, LIST_AN, z_row, size_batch):
    hET_A_z_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = hE_row[i,:]  
        A_1_Example = LIST_AN[i]
        z_col_1_Example = z_row[i,:].T
        hET_A = np.matmul( hE_row_1_Example, A_1_Example )
        hET_A_z = np.matmul(hET_A, z_col_1_Example) 
        hET_A_z_Multiple_Examples.append(hET_A_z)
    return np.reshape(hET_A_z_Multiple_Examples, [size_batch,1])

def np__squaredNorm_hET_A_z(hET_A_z, size_batch):
    hET_A_z_2 = np.linalg.norm(hET_A_z,axis=1)**2  
    return np.reshape(hET_A_z_2, [size_batch,1]) 

def np__hET_b(hE_row, LIST_beamforming_col, size_batch):
    hET_beam_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = hE_row[i,:]
        beam_col_1_Example = LIST_beamforming_col[i]
        hET_beam = np.matmul(hE_row_1_Example, beam_col_1_Example)
        hET_beam_Multiple_Examples.append(hET_beam)
    return np.reshape(hET_beam_Multiple_Examples, [size_batch,1])

def np__squaredNorm_hET_A(hE_row, LIST_AN, size_batch):
    squaredNorm_hET_A_Multiple_Examples = []
    for i in range(size_batch):
        hE_row_1_Example = hE_row[i,:]   
        A_1_Example = LIST_AN[i]
        hET_A = np.matmul( hE_row_1_Example, A_1_Example )
        squaredNorm_hET_A = np.linalg.norm(hET_A)**2
        squaredNorm_hET_A_Multiple_Examples.append(squaredNorm_hET_A)
    return np.reshape(squaredNorm_hET_A_Multiple_Examples, [size_batch,1])

def np__squaredNorm_of_a_complex_vector(u_row, size_batch):
    u_row_conj = u_row.conj() 
    product_elementwise = np.multiply(u_row, u_row_conj)
    product_elementwise = np.real( product_elementwise ) #tf.real( a + 0j ) = a 
    squaredNorm_of_u = np.sum(product_elementwise, axis=1)
    return np.reshape(squaredNorm_of_u, [size_batch,1])

def np__squaredNorm_nB(nB, size_batch):
    nB_2 = np.linalg.norm(nB,axis=1)**2  
    return np.reshape(nB_2, [size_batch,1]) 

def np_Norm_hB(hB_row, size_batch):
    Norm_hB_Multiple_Examples = []
    for i in range(size_batch):
        Norm_hB_1_Example = np.linalg.norm(hB_row[i])
        Norm_hB_Multiple_Examples.append(Norm_hB_1_Example)
    return np.reshape(Norm_hB_Multiple_Examples, [size_batch,1])


''' Operations on real variables '''
def Norm_of_a_real_vector(u_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    norm_of_vector = tf.norm(u_row, axis=1)
    # norm_of_vector = [ [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ]   ex1
    #                    [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ]   ex2
    #                       ..................................
    #                    [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ] ] exN
    return tf.reshape(norm_of_vector, [tf.size(norm_of_vector),1])

def squaredNorm_of_a_real_vector(u_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    squared_u_element_wise = u_row**2
    # squared_u_element_wise = [ [u_1**2, u_2**2, ... , u_N**2]     example 1
    #                            [u_1**2, u_2**2, ... , u_N**2]     example 2
    #                             ... ,    ... ,  ... , ...
    #                            [u_1**2, u_2**2, ... , u_N**2] ]   example ?
    squaredNorm_of_vector = tf.reduce_sum(squared_u_element_wise, 
                                          axis=1, keepdims=True)
    #squaredNorm_of_vector = [ [u_1**2 + u_2**2 + ... + u_N**2]   example 1
    #                          [u_1**2 + u_2**2 + ... + u_N**2]   example 2
    #                           ..............................
    #                          [u_1**2 + u_2**2 + ... + u_N**2]   example ?
    return squaredNorm_of_vector

def Norm_of_real_uT_times_v(u_row, v_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    #################################################
    # v_row = [ [v_1, v_2, ... , v_N]     example 1
    #           [v_1, v_2, ... , v_N]     example 2
    #            ..., ..., ... , ...
    #           [v_1, v_2, ... , v_N] ]   example ?
    #################################################
    uv_element_wise = tf.multiply(u_row,v_row)
    # uv_element_wise = [ [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 1
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 2
    #                      ..., ..., ... , ...
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N] ]   example ?
    #################################################
    uH_times_v = tf.reduce_sum(uv_element_wise, axis=1, keepdims=True)
    # uH_times_v = [ [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 1
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 2
    #                ..................................
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N] ]   example ?
    # shape [?, 1] where ? is the size of a batch
    #################################################
    uH_times_v_norm = tf.norm(uH_times_v) 
    # squaredNorm = [ [ a_1**2 ]     example 1
    #                 [ a_2**2 ]     example 2
    #                  ...
    #                 [ a_N**2 ] ]   example ?
    return uH_times_v_norm

def squaredNorm_of_real_uT_times_v(u_row, v_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    #################################################
    # v_row = [ [v_1, v_2, ... , v_N]     example 1
    #           [v_1, v_2, ... , v_N]     example 2
    #           ..., ..., ... , ...
    #           [v_1, v_2, ... , v_N] ]   example ?
    #################################################
    uv_element_wise = tf.multiply(u_row,v_row)
    # uv_element_wise = [ [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 1
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 2
    #                      ..., ..., ... , ...
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N] ]   example ?
    #################################################
    uH_times_v = tf.reduce_sum(uv_element_wise, axis=1, keepdims=True)
    # uH_times_v = [ [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 1
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 2
    #                ..................................
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N] ]   example ?
    # shape [?, 1] where ? is the size of a batch
    #################################################
    squaredNorm = uH_times_v**2 
    # squaredNorm = [ [ a_1**2 ]     example 1
    #                 [ a_2**2 ]     example 2
    #                  ...
    #                 [ a_N**2 ] ]   example ?
    return squaredNorm

def Norm_of_real_xT_times_A(x_row,A):
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xH_times_Ax_row_vector = tf.matmul(x_row,A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    xA_norm = tf.norm(xH_times_Ax_row_vector,axis=1)
    return tf.reshape(xA_norm, [tf.size(xA_norm),1])

def squaredNorm_of_real_xT_times_A(x_row,A):
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xH_times_Ax_row_vector = tf.matmul(x_row,A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    squared_xH_times_A_element_wise = tf.abs(xH_times_Ax_row_vector)**2
    #squared_xH_times_A_element_wise = [ [h_1**2, h_2**2, ..., h_N**2]    example 1
    #                                    [h_1**2, h_2**2, ..., h_N**2]    example 2
    #                                      ...  ,  ... ,  ...,  ...
    #                                    [h_1**2, h_2**2, ..., h_N**2] ]  example ?
    #################################################
    squaredNorm = tf.reduce_sum(squared_xH_times_A_element_wise, axis=1)
    # squaredNorm = [ [h_1**2 + h_2**2 + ...+ h_N**2]    example 1
    #                 [h_1**2 + h_2**2 + ...+ h_N**2]    example 2
    #                   .........................
    #                 [h_1**2 + h_2**2 + ...+ h_N**2] ]  example ?
    #
    #             = [ [ ||h||^2 ]     example 1
    #                 [ ||h||^2 ]     example 2
    #                 ..........
    #                 [ ||h||^2 ] ]   example ?
    return tf.reshape(squaredNorm, [tf.size(squaredNorm),1])
    