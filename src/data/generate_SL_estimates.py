import numpy as np
import scipy
import math
import click



import scipy.optimize as op
from scipy.optimize import Bounds
import time

import scipy.io as sio

from eit import EIT
from fem import Mesh, V_h, dtn_map



def generate_GCOORD(lx, ly, nx, ny):

    x_coords = np.linspace(-lx / 2, lx / 2, nx)
    y_coords = np.linspace(-ly / 2, ly / 2, ny)
    
    xv, yv = np.meshgrid(x_coords, y_coords)
    GCOORD = np.vstack([xv.ravel(), yv.ravel()]).T
    
    return GCOORD

def assemble_EL_connectivity(nel, nnodel, nex, nx):
    EL2NOD = np.zeros((nel,nnodel), dtype=int)

    for iel in range(0,nel):
        row = iel//nex   
        ind = iel + row
        EL2NOD[iel,:] = np.array([ind, ind+1, ind+nx+1, ind+nx])
        
    return EL2NOD

def interpolate_pts(known_pts, known_vals, interp_pts):

    interp_vals = scipy.interpolate.griddata(known_pts, known_vals, interp_pts, method='linear', fill_value=1.)

    for i in range(len(interp_pts)):
        curr_pt = interp_pts[i]
        dist = math.sqrt(curr_pt[0]**2 + curr_pt[1]**2)
        if dist >= 1:
            interp_vals[i] = 1.

    return interp_vals

def generate_EIT_sol(num_iters, p, t, bdy_idx, vol_idx, sigma_vec_true, noise):

    # define the mesh
    mesh = Mesh(p, t, bdy_idx, vol_idx)

    # define the approximation space
    v_h = V_h(mesh)

    # extracting the DtN data
    dtn_data, sol = dtn_map(v_h, sigma_vec_true)

    noise_data = noise * dtn_data

    dtn_data = dtn_data + noise_data

    # this is the initial guess
    sigma_vec_0 = 1. + np.zeros(t.shape[0], dtype=np.float64)

    # we create the eit wrapper
    eit = EIT(v_h)

    # build the stiffness matrices
    eit.update_matrices(sigma_vec_0)

    def J(x):
        return eit.misfit(dtn_data, x)
    
    opt_tol = 1e-30

    bounds_l = [1. for _ in range(len(sigma_vec_0))]
    bounds_r = [np.inf for _ in range(len(sigma_vec_0))]
    bounds = Bounds(bounds_l, bounds_r)

    # t_i = time.time()
    res = op.minimize(J, sigma_vec_0, method='L-BFGS-B',
                      jac = True,
                      tol = opt_tol,
                      bounds=bounds, 
                      options={'maxiter': num_iters,
                                'disp': False, 'ftol':opt_tol, 'gtol':opt_tol}, 
                     )
                       # callback=callback)

    # t_f = time.time()
    # print(f'Time elapsed is {(t_f - t_i):.4f}', flush=True)

    return res.x

@click.command()
@click.option('--img-size', type=int, required=True, help='size of output image')
@click.option('--noise', type=float, required=True, help='noise level')
@click.option('--num-iters', type=int, required=True, help='max number of BFGS iterations')
@click.option('--start', type=int, required=True, help='start number')
@click.option('--end', type=int, required=True, help='end number')
@click.option('--mesh-path', type=str, required=True, help='path to the mesh file')
@click.option('--dataset-path', type=str, required=True, help='directory with MAF datasets')
@click.option('--new-data-path', type=str, required=True, help='path to the model checkpoint')
@click.option('--load-path', type=str, required=False, help='path to stored data')
def main(
    img_size: int,
    noise: float,
    num_iters: int,
    start: int,
    end: int,
    mesh_path: str,
    dataset_path: str,
    new_data_path: str,
    load_path: str,
):

    #geometry
    nx          = img_size + 1
    ny          = img_size + 1
    lx          = 2
    ly          = 2
    nnodel      = 4  #number of nodes per element
    
    # model parameters
    nex         = nx-1
    ney         = ny-1
    nnod        = nx*ny #number of nodes
    nel         = nex*ney #number of finite elements

     #generate square mesh and element connectivity
    GCOORD = generate_GCOORD(lx, ly, nx, ny)
    EL2NOD = assemble_EL_connectivity(nnod, nnodel, nex, nx)

    #retrieve construction of circular mesh
    mat_fname  = mesh_path
    mat_contents = sio.loadmat(mat_fname)
    
    # points
    p = np.array(mat_contents['p'])
    #triangle
    t = np.array(mat_contents['t']-1) # all the indices should be reduced by one
    # volumetric indices
    vol_idx = mat_contents['vol_idx'].reshape((-1,))-1 # all the indices should be reduced by one
    # indices at the boundaries
    bdy_idx = mat_contents['bdy_idx'].reshape((-1,))-1 # all the indices should be reduced by one
    
    centroids = np.mean(p[t], axis=1)  # Calculate the mean along the columns to get the centroids
    print("here")
    
    path = dataset_path
    data_arr = np.load(path)

    num_samples = len(data_arr['sigma'])
    
    sigma_true = np.zeros((num_samples, len(centroids)))
    sigma_pred = np.zeros((num_samples, len(centroids)))
    imgs_true = np.zeros((num_samples, nx-1, ny-1))
    imgs_pred = np.zeros((num_samples, nx-1, ny-1))

    start_num = start
    print(start_num, flush=True)
    print(nx, ny, flush=True)
    print(imgs_true.shape, flush=True)

    if start_num != 0: 
        curr_ims = np.load(load_path)
        print(curr_ims['imgs_true'].shape)
        print(np.count_nonzero(curr_ims['imgs_true']))
        sigma_true[:start_num, ...] = curr_ims['sigma_true'][:start_num, ...]
        sigma_pred[:start_num, ...] = curr_ims['sigma_pred'][:start_num, ...]
        imgs_true[:start_num, ...] = curr_ims['imgs_true'][:start_num, ...]
        imgs_pred[:start_num, ...] = curr_ims['imgs_pred'][:start_num, ...]
    # print(len(data_arr))
    print(np.count_nonzero(imgs_pred), flush=True)

    save_path = new_data_path + '_i_' + str(num_iters) + '_r_' + str(img_size) + '_n_' + str(noise) 
    
    for i in range(start_num, end):
        sigma_vec_true = data_arr['sigma'][i]

        t_i = time.time()
        sigma_vec_pred = generate_EIT_sol(num_iters, p, t, bdy_idx, vol_idx, sigma_vec_true, noise)
        
        sq_img_true = 1. + np.zeros((nx-1) * (ny-1))
        sq_img_pred = 1. + np.zeros((nx-1) * (ny-1))

         
        interp_vals_true = interpolate_pts(centroids, sigma_vec_true, GCOORD)
        interp_vals_pred = interpolate_pts(centroids, sigma_vec_pred, GCOORD)
        for iel in range(0,nel):
            # ECOORD = np.take(GCOORD, EL2NOD[iel, :], axis=0)
            ECOORD_true = np.take(interp_vals_true, EL2NOD[iel, :], axis=0)
            ECOORD_pred = np.take(interp_vals_pred, EL2NOD[iel, :], axis=0)
            # print(ECOORD)
            #based on ECOORD pts, average them out to find pixel value and assing
            
            sq_img_true[iel] = np.mean(ECOORD_true)
            sq_img_pred[iel] = np.mean(ECOORD_pred)
            
            # break
        t_f = time.time()
        # sigma_vec_0 = res.x
        
    
        sq_img_true = np.flip(sq_img_true.reshape((nx-1, ny-1)), axis=0)
        sq_img_pred = np.flip(sq_img_pred.reshape((nx-1, ny-1)), axis=0)
        
        sigma_true[i, ...] = sigma_vec_true
        sigma_pred[i, ...] = sigma_vec_pred
        imgs_true[i, ...] = sq_img_true
        imgs_pred[i, ...] = sq_img_pred
        
        if i % 100 == 0:
            print(f'Time elapsed is {(t_f - t_i):.4f}', flush=True)
            print(i, flush=True)
            npy_name = save_path
            np.savez(npy_name, imgs_true=imgs_true, imgs_pred=imgs_pred, sigma_true=sigma_true, sigma_pred=sigma_pred)
        
        #append to big new dataset here
        
    #save big new dataset when done
    print(imgs_true.shape)
    print(imgs_pred.shape)
    
    npy_name = save_path
    np.savez(npy_name, imgs_true=imgs_true, imgs_pred=imgs_pred, sigma_true=sigma_true, sigma_pred=sigma_pred)
    
    
        

# Check if the script is being run directly (not imported)
if __name__ == "__main__":
    main()
    



