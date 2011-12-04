/***************************************************************************
                                lal_eam.cpp
                             -------------------
                      W. Michael Brown, Trung Dac Nguyen (ORNL)

  Class for acceleration of the eam pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : 
    email                : brownw@ornl.gov nguyentd@ornl.gov
 ***************************************************************************/
 
#ifdef USE_OPENCL
#include "eam_cl.h"
#else
#include "eam_ptx.h"
#endif

#include "lal_eam.h"
#include <cassert>
using namespace LAMMPS_AL;
#define EAMT EAM<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
EAMT::EAM() : BaseAtomic<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
EAMT::~EAM() {
  clear();
}
 
template <class numtyp, class acctyp>
int EAMT::init(const int ntypes, double host_cutforcesq,
              int **host_type2rhor, int **host_type2z2r, int *host_type2frho,
              double ***host_rhor_spline, double ***host_z2r_spline,
              double ***host_frho_spline,
              double rdr, double rdrho, int nrhor, int nrho, 
              int nz2r, int nfrho, int nr,
              const int nlocal, const int nall, const int max_nbors,
              const int maxspecial, const double cell_size,
              const double gpu_split, FILE *_screen) 
{
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,eam);
  
  if (success!=0)
    return success;
  
  // allocate fp
  
  bool cpuview=false;
  if (this->ucl_device->device_type()==UCL_CPU)
    cpuview=true;
  
  _max_fp_size=static_cast<int>(static_cast<double>(nall)*1.10);
  host_fp.alloc(_max_fp_size,*(this->ucl_device));
  if (cpuview)
    dev_fp.view(host_fp);
  else 
    dev_fp.alloc(_max_fp_size,*(this->ucl_device),UCL_WRITE_ONLY);
                                     
  k_energy.set_function(*(this->pair_program),"kernel_energy");
  k_energy_fast.set_function(*(this->pair_program),"kernel_energy_fast");
  fp_tex.get_texture(*(this->pair_program),"fp_tex");
  fp_tex.bind_float(dev_fp,1);

  // Initialize timers for selected GPU
  time_pair2.init(*(this->ucl_device));
  time_pair2.zero();
  
  time_fp1.init(*(this->ucl_device));
  time_fp1.zero();
  
  time_fp2.init(*(this->ucl_device));
  time_fp2.zero();

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;

  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  
  _ntypes=lj_types;
  _cutforcesq=host_cutforcesq;
  _rdr=rdr;
  _rdrho = rdrho;
  _nrhor=nrhor;
  _nrho=nrho;
  _nz2r=nz2r;
  _nfrho=nfrho;
  _nr=nr;
  
  UCL_H_Vec<numtyp> dview_type(lj_types*lj_types*2,*(this->ucl_device),
                               UCL_WRITE_OPTIMIZED);
  
  for (int i=0; i<lj_types*lj_types*2; i++)
    dview_type[i]=(numtyp)0.0; 
                                
  // pack type2rhor and type2z2r
  type2rhor_z2r.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  
  this->atom->type_pack2(ntypes,lj_types,type2rhor_z2r,dview_type,
                        host_type2rhor,
                        host_type2z2r);
  
  // pack type2frho
  UCL_H_Vec<numtyp> dview_type2frho(ntypes,*(this->ucl_device),
                               UCL_WRITE_OPTIMIZED);

  type2frho.alloc(ntypes,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0; i<ntypes; i++)
    dview_type2frho[i]=(numtyp)host_type2frho[i];
  ucl_copy(type2frho,dview_type2frho,false);
                        
  // pack frho_spline
  UCL_H_Vec<numtyp4> dview_frho_spline(nfrho*(nr+1),*(this->ucl_device),
                               UCL_WRITE_OPTIMIZED);
                               
  for (int ix=0; ix<nfrho; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_frho_spline[ix*(nr+1)+iy].x=host_frho_spline[ix][iy][0];
    dview_frho_spline[ix*(nr+1)+iy].y=host_frho_spline[ix][iy][1];
    dview_frho_spline[ix*(nr+1)+iy].z=host_frho_spline[ix][iy][2];
    dview_frho_spline[ix*(nr+1)+iy].w=0;
  }

  frho_spline1.alloc(nfrho*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(frho_spline1,dview_frho_spline,false);
  frho_spline1_tex.get_texture(*(this->pair_program),"frho_sp1_tex");
  frho_spline1_tex.bind_float(frho_spline1,4);

  for (int ix=0; ix<nfrho; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_frho_spline[ix*(nr+1)+iy].x=host_frho_spline[ix][iy][3];
    dview_frho_spline[ix*(nr+1)+iy].y=host_frho_spline[ix][iy][4];
    dview_frho_spline[ix*(nr+1)+iy].z=host_frho_spline[ix][iy][5];
    dview_frho_spline[ix*(nr+1)+iy].w=host_frho_spline[ix][iy][6];
  }

  frho_spline2.alloc(nfrho*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(frho_spline2,dview_frho_spline,false);
  frho_spline2_tex.get_texture(*(this->pair_program),"frho_sp2_tex");
  frho_spline2_tex.bind_float(frho_spline2,4);

  // pack rhor_spline
  UCL_H_Vec<numtyp4> dview_rhor_spline(nrhor*(nr+1),*(this->ucl_device),
                               UCL_WRITE_OPTIMIZED);
                               
  for (int ix=0; ix<nrhor; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_rhor_spline[ix*(nr+1)+iy].x=host_rhor_spline[ix][iy][0];
    dview_rhor_spline[ix*(nr+1)+iy].y=host_rhor_spline[ix][iy][1];
    dview_rhor_spline[ix*(nr+1)+iy].z=host_rhor_spline[ix][iy][2];
    dview_rhor_spline[ix*(nr+1)+iy].w=(numtyp)0;
  }

  rhor_spline1.alloc(nrhor*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(rhor_spline1,dview_rhor_spline,false);
  rhor_spline1_tex.get_texture(*(this->pair_program),"rhor_sp1_tex");
  rhor_spline1_tex.bind_float(rhor_spline1,4);

  for (int ix=0; ix<nrhor; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_rhor_spline[ix*(nr+1)+iy].x=host_rhor_spline[ix][iy][3];
    dview_rhor_spline[ix*(nr+1)+iy].y=host_rhor_spline[ix][iy][4];
    dview_rhor_spline[ix*(nr+1)+iy].z=host_rhor_spline[ix][iy][5];
    dview_rhor_spline[ix*(nr+1)+iy].w=host_rhor_spline[ix][iy][6];
  }

  rhor_spline2.alloc(nrhor*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(rhor_spline2,dview_rhor_spline,false);
  rhor_spline2_tex.get_texture(*(this->pair_program),"rhor_sp2_tex");
  rhor_spline2_tex.bind_float(rhor_spline2,4);

  // pack z2r_spline
  UCL_H_Vec<numtyp4> dview_z2r_spline(nz2r*(nr+1),*(this->ucl_device),
                               UCL_WRITE_OPTIMIZED);
                               
  for (int ix=0; ix<nz2r; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_z2r_spline[ix*(nr+1)+iy].x=host_z2r_spline[ix][iy][0];
    dview_z2r_spline[ix*(nr+1)+iy].y=host_z2r_spline[ix][iy][1];
    dview_z2r_spline[ix*(nr+1)+iy].z=host_z2r_spline[ix][iy][2];
    dview_z2r_spline[ix*(nr+1)+iy].w=(numtyp)0;
  }
  
  z2r_spline1.alloc(nz2r*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(z2r_spline1,dview_z2r_spline,false);
  z2r_spline1_tex.get_texture(*(this->pair_program),"z2r_sp1_tex");
  z2r_spline1_tex.bind_float(z2r_spline1,4);
  
  for (int ix=0; ix<nz2r; ix++)
    for (int iy=0; iy<nr+1; iy++) {
    dview_z2r_spline[ix*(nr+1)+iy].x=host_z2r_spline[ix][iy][3];
    dview_z2r_spline[ix*(nr+1)+iy].y=host_z2r_spline[ix][iy][4];
    dview_z2r_spline[ix*(nr+1)+iy].z=host_z2r_spline[ix][iy][5];
    dview_z2r_spline[ix*(nr+1)+iy].w=host_z2r_spline[ix][iy][6];
  }
  
  z2r_spline2.alloc(nz2r*(nr+1),*(this->ucl_device),UCL_READ_ONLY);
  ucl_copy(z2r_spline2,dview_z2r_spline,false);
  z2r_spline2_tex.get_texture(*(this->pair_program),"z2r_sp2_tex");
  z2r_spline2_tex.bind_float(z2r_spline2,4);

  _allocated=true;
  this->_max_bytes=type2rhor_z2r.row_bytes()
        + type2frho.row_bytes()
        + rhor_spline1.row_bytes()
        + rhor_spline2.row_bytes()
        + frho_spline1.row_bytes()
        + frho_spline2.row_bytes()
        + z2r_spline1.row_bytes()
        + z2r_spline2.row_bytes()
        + dev_fp.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void EAMT::clear() {
  if (!_allocated)
    return;
  _allocated=false;
  
  type2rhor_z2r.clear();
  type2frho.clear();
  rhor_spline1.clear();
  rhor_spline2.clear();
  frho_spline1.clear();
  frho_spline2.clear();
  z2r_spline1.clear();
  z2r_spline2.clear();
  
  host_fp.clear();
  dev_fp.clear();
  
  time_pair2.clear();
  time_fp1.clear();
  time_fp2.clear();
  
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double EAMT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(EAM<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Copy nbor list from host if necessary and then compute atom energies/forces
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void EAMT::compute(const int f_ago, const int inum_full,
                   const int nall, double **host_x, int *host_type,
                   int *ilist, int *numj, int **firstneigh,
                   const bool eflag, const bool vflag,
                   const bool eatom, const bool vatom,
                   int &host_start, const double cpu_time,
                   bool &success, void **fp_ptr) {
  this->acc_timers();
  
  if (this->device->time_device()) {
    // Put time from the second part to the total time_pair
    this->time_pair.add_time_to_total(time_pair2.time());
    
    // Add transfer time from device -> host after part 1
    this->atom->add_transfer_time(time_fp1.time());
    
    // Add transfer time from host -> device before part 2
    this->atom->add_transfer_time(time_fp2.time());
  }
  
  if (inum_full==0) {
    host_start=0;
    // Make sure textures are correct if realloc by a different hybrid style
    this->resize_atom(0,nall,success);
    this->zero_timers();
    return;
  }
  
  int ago=this->hd_balancer.ago_first(f_ago);
  int inum=this->hd_balancer.balance(ago,inum_full,cpu_time);
  this->ans->inum(inum);
  host_start=inum;

  // ------------------- Resize FP Array for EAM --------------------
  
  if (nall>_max_fp_size) {
    dev_fp.clear();
    host_fp.clear();
    
    _max_fp_size=static_cast<int>(static_cast<double>(nall)*1.10);
    host_fp.alloc(_max_fp_size,*(this->ucl_device));
    if (this->ucl_device->device_type()==UCL_CPU)
      dev_fp.view(host_fp);
    else 
      dev_fp.alloc(_max_fp_size,*(this->ucl_device),UCL_WRITE_ONLY);
    
    fp_tex.bind_float(dev_fp,1);
  }
  *fp_ptr=host_fp.begin();  

  // -----------------------------------------------------------------

  if (ago==0) {
    this->reset_nbors(nall, inum, ilist, numj, firstneigh, success);
    if (!success)
      return;
  }
  
  this->atom->cast_x_data(host_x,host_type);
  this->atom->add_x_data(host_x,host_type);

  loop(eflag,vflag);
  
  // copy fp from device to host for comm
  time_fp1.start();
  ucl_copy(host_fp,dev_fp,false);
  time_fp1.stop();
}

// ---------------------------------------------------------------------------
// Reneighbor on GPU and then compute per-atom densities
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int** EAMT::compute(const int ago, const int inum_full,
                    const int nall, double **host_x, int *host_type,
                    double *sublo, double *subhi, int *tag,
                    int **nspecial, int **special, const bool eflag, 
                    const bool vflag, const bool eatom,
                    const bool vatom, int &host_start,
                    int **ilist, int **jnum,
                    const double cpu_time, bool &success,
                    int &inum, void **fp_ptr) {
  this->acc_timers();
  
  if (this->device->time_device()) {
    // Put time from the second part to the total time_pair
    this->time_pair.add_time_to_total(time_pair2.time());
    
    // Add transfer time from device -> host after part 1
    this->atom->add_transfer_time(time_fp1.time());
    
    // Add transfer time from host -> device before part 2
    this->atom->add_transfer_time(time_fp2.time());
  }
  
  if (inum_full==0) {
    host_start=0;
    // Make sure textures are correct if realloc by a different hybrid style
    this->resize_atom(0,nall,success);
    this->zero_timers();
    return NULL;
  }
  
  // load balance, returning the atom count on the device (inum)
  this->hd_balancer.balance(cpu_time);
  inum=this->hd_balancer.get_gpu_count(ago,inum_full);
  this->ans->inum(inum);
  host_start=inum;
 
  // ------------------- Resize FP Array for EAM --------------------
  
  if (nall>_max_fp_size) {
    dev_fp.clear();
    host_fp.clear();
    
    _max_fp_size=static_cast<int>(static_cast<double>(nall)*1.10);
    host_fp.alloc(_max_fp_size,*(this->ucl_device));
    if (this->ucl_device->device_type()==UCL_CPU)
      dev_fp.view(host_fp);
    else 
      dev_fp.alloc(_max_fp_size,*(this->ucl_device),UCL_WRITE_ONLY);
    
    fp_tex.bind_float(dev_fp,1);
  }      
  *fp_ptr=host_fp.begin();  

  // -----------------------------------------------------------------

  // Build neighbor list on GPU if necessary 
  if (ago==0) {
    this->build_nbor_list(inum, inum_full-inum, nall, host_x, host_type,
                    sublo, subhi, tag, nspecial, special, success);
    if (!success)
      return NULL;
  } else {
    this->atom->cast_x_data(host_x,host_type);
    this->atom->add_x_data(host_x,host_type);
  }
  *ilist=this->nbor->host_ilist.begin();
  *jnum=this->nbor->host_acc.begin();

  loop(eflag,vflag);
  
  // copy fp from device to host for comm
  time_fp1.start();
  ucl_copy(host_fp,dev_fp,false);
  time_fp1.stop();
  
  return this->nbor->host_jlist.begin()-host_start;
}

// ---------------------------------------------------------------------------
// Copy nbor list from host if necessary and then calculate forces, virials,..
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void EAMT::compute2(int *ilist, const bool eflag, const bool vflag,
                    const bool eatom, const bool vatom) {
  this->hd_balancer.start_timer();
  time_fp2.start();
  this->add_fp_data();
  time_fp2.stop();
  
  loop2(eflag,vflag);
  if (ilist == NULL)
    this->ans->copy_answers(eflag,vflag,eatom,vatom);
  else
    this->ans->copy_answers(eflag,vflag,eatom,vatom, ilist);
  
  this->device->add_ans_object(this->ans);
  this->hd_balancer.stop_timer();
}

// ---------------------------------------------------------------------------
// Calculate per-atom energies and forces
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void EAMT::loop(const bool _eflag, const bool _vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int eflag, vflag;
  if (_eflag)
    eflag=1;
  else
    eflag=0;

  if (_vflag)
    vflag=1;
  else
    vflag=0;
  
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  
  if (shared_types) {
    this->k_energy_fast.set_size(GX,BX);
    this->k_energy_fast.run(&this->atom->dev_x.begin(), 
                 &type2rhor_z2r.begin(), &type2frho.begin(),
                 &rhor_spline2.begin(),
                 &frho_spline1.begin(),&frho_spline2.begin(), 
                 &this->nbor->dev_nbor.begin(), &this->_nbor_data->begin(),
                 &dev_fp.begin(), 
                 &this->ans->dev_engv.begin(),
                 &eflag, &ainum,
                 &nbor_pitch, 
                 &_ntypes, &_cutforcesq, 
                 &_rdr, &_rdrho,
                 &_nrho, &_nr,
                 &this->_threads_per_atom);
  }
  else {
    this->k_energy.set_size(GX,BX);
    this->k_energy.run(&this->atom->dev_x.begin(), 
                 &type2rhor_z2r.begin(), &type2frho.begin(),
                 &rhor_spline2.begin(),
                 &frho_spline1.begin(),&frho_spline2.begin(), 
                 &this->nbor->dev_nbor.begin(), &this->_nbor_data->begin(),
                 &dev_fp.begin(), 
                 &this->ans->dev_engv.begin(),
                 &eflag, &ainum,
                 &nbor_pitch, 
                 &_ntypes, &_cutforcesq, 
                 &_rdr, &_rdrho,
                 &_nrho, &_nr,
                 &this->_threads_per_atom);
  }

  this->time_pair.stop();
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void EAMT::loop2(const bool _eflag, const bool _vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int eflag, vflag;
  if (_eflag)
    eflag=1;
  else
    eflag=0;

  if (_vflag)
    vflag=1;
  else
    vflag=0;
  
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair2.start();
  
  if (shared_types) {
    this->k_pair_fast.set_size(GX,BX);
    this->k_pair_fast.run(&this->atom->dev_x.begin(), &dev_fp.begin(), 
                   &type2rhor_z2r.begin(),
                   &rhor_spline1.begin(), 
                   &z2r_spline1.begin(),
                   &z2r_spline2.begin(), 
                   &this->nbor->dev_nbor.begin(),
                   &this->_nbor_data->begin(), &this->ans->dev_ans.begin(),
                   &this->ans->dev_engv.begin(), &eflag, &vflag, &ainum,
                   &nbor_pitch, &_cutforcesq, &_rdr, &_nr,
                   &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->dev_x.begin(), &dev_fp.begin(), 
                   &type2rhor_z2r.begin(),
                   &rhor_spline1.begin(), 
                   &z2r_spline1.begin(),
                   &z2r_spline2.begin(),
                   &this->nbor->dev_nbor.begin(),
                   &this->_nbor_data->begin(), &this->ans->dev_ans.begin(),
                   &this->ans->dev_engv.begin(), &eflag, &vflag, &ainum,
                   &nbor_pitch, &_ntypes, &_cutforcesq, &_rdr, &_nr,
                   &this->_threads_per_atom);
  }

  this->time_pair2.stop();
}

template class EAM<PRECISION,ACC_PRECISION>;
