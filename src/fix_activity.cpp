/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
 
 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.
 
 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 Contributing authors: Carolyn Phillips (U Mich), reservoir energy tally
 Aidan Thompson (SNL) GJF formulation
 ------------------------------------------------------------------------- */

#include <iostream>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fix_activity.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};

#define SINERTIA 0.4          // moment of inertia prefactor for sphere
#define EINERTIA 0.2          // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

FixActivity::FixActivity(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg),
gjfflag(0), gfactor1(NULL), gfactor2(NULL), ratio(NULL), tstr(NULL),
factivity(NULL), tforce(NULL), franprev(NULL), id_temp(NULL), random(NULL)
{
    if (narg < 8) error->all(FLERR,"Illegal fix activity command");
    
    dynamic_group_allow = 1;
    scalar_flag = 1;
    global_freq = 1;
    extscalar = 1;
    nevery = 1;
    
    if (strstr(arg[3],"v_") == arg[3]) {
        int n = strlen(&arg[3][2]) + 1;
        tstr = new char[n];
        strcpy(tstr,&arg[3][2]);
    } else {
        t_start = force->numeric(FLERR,arg[3]);
        t_target = t_start;
        tstyle = CONSTANT;
    }
    
    t_stop = force->numeric(FLERR,arg[4]);
    t_period = force->numeric(FLERR,arg[5]);
    seed = force->inumeric(FLERR,arg[6]);
    // This should allow me to avoid rebuilding every time I need to edit the activity
    peclet = force->numeric(FLERR,arg[7]);
    
    if (t_period <= 0.0) error->all(FLERR,"Fix activity period must be > 0.0");
    if (seed <= 0) error->all(FLERR,"Illegal fix activity command");
    
    // initialize Marsaglia RNG with processor-unique seed
    
    random = new RanMars(lmp,seed + comm->me);
    
    // allocate per-type arrays for force prefactors
    
    gfactor1 = new double[atom->ntypes+1];
    gfactor2 = new double[atom->ntypes+1];
    ratio = new double[atom->ntypes+1];
    
    // optional args
    
    for (int i = 1; i <= atom->ntypes; i++) ratio[i] = 1.0;
    ascale = 0.0;
    gjfflag = 0;
    oflag = 0;
    tallyflag = 0;
    zeroflag = 0;
    
    int iarg = 8;
    while (iarg < narg) {
        if (strcmp(arg[iarg],"angmom") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal fix activity command");
            if (strcmp(arg[iarg+1],"no") == 0) ascale = 0.0;
            else ascale = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        } else if (strcmp(arg[iarg],"gjf") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal fix activity command");
            if (strcmp(arg[iarg+1],"no") == 0) gjfflag = 0;
            else if (strcmp(arg[iarg+1],"yes") == 0) gjfflag = 1;
            else error->all(FLERR,"Illegal fix activity command");
            iarg += 2;
        } else if (strcmp(arg[iarg],"omega") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal fix activity command");
            if (strcmp(arg[iarg+1],"no") == 0) oflag = 0;
            else if (strcmp(arg[iarg+1],"yes") == 0) oflag = 1;
            else error->all(FLERR,"Illegal fix activity command");
            iarg += 2;
        } else if (strcmp(arg[iarg],"scale") == 0) {
            if (iarg+3 > narg) error->all(FLERR,"Illegal fix activity command");
            int itype = force->inumeric(FLERR,arg[iarg+1]);
            double scale = force->numeric(FLERR,arg[iarg+2]);
            if (itype <= 0 || itype > atom->ntypes)
                error->all(FLERR,"Illegal fix activity command");
            ratio[itype] = scale;
            iarg += 3;
        } else if (strcmp(arg[iarg],"tally") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal fix activity command");
            if (strcmp(arg[iarg+1],"no") == 0) tallyflag = 0;
            else if (strcmp(arg[iarg+1],"yes") == 0) tallyflag = 1;
            else error->all(FLERR,"Illegal fix activity command");
            iarg += 2;
        } else if (strcmp(arg[iarg],"zero") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal fix activity command");
            if (strcmp(arg[iarg+1],"no") == 0) zeroflag = 0;
            else if (strcmp(arg[iarg+1],"yes") == 0) zeroflag = 1;
            else error->all(FLERR,"Illegal fix activity command");
            iarg += 2;
        } else error->all(FLERR,"Illegal fix activity command");
    }
    
    // set temperature = NULL, user can override via fix_modify if wants bias
    
    id_temp = NULL;
    temperature = NULL;
    
    energy = 0.0;
    
    // factivity is unallocated until first call to setup()
    // compute_scalar checks for this and returns 0.0
    // if factivity_allocated is not set
    
    factivity = NULL;
    factivity_allocated = 0;
    franprev = NULL;
    tforce = NULL;
    maxatom1 = maxatom2 = 0;
    
    // Setup atom-based array for franprev
    // register with Atom class
    // No need to set peratom_flag
    // as this data is for internal use only
    
    if (gjfflag) {
        nvalues = 3;
        grow_arrays(atom->nmax);
        atom->add_callback(0);
        
        // initialize franprev to zero
        
        int nlocal = atom->nlocal;
        for (int i = 0; i < nlocal; i++) {
            franprev[i][0] = 0.0;
            franprev[i][1] = 0.0;
            franprev[i][2] = 0.0;
        }
    }
    
    if (tallyflag && zeroflag && comm->me == 0)
        error->warning(FLERR,"Energy tally does not account for 'zero yes'");
}

/* ---------------------------------------------------------------------- */

FixActivity::~FixActivity()
{
    delete random;
    delete [] tstr;
    delete [] gfactor1;
    delete [] gfactor2;
    delete [] ratio;
    delete [] id_temp;
    memory->destroy(factivity);
    memory->destroy(tforce);
    
    if (gjfflag) {
        memory->destroy(franprev);
        atom->delete_callback(id,0);
    }
}

/* ---------------------------------------------------------------------- */

int FixActivity::setmask()
{
    int mask = 0;
    mask |= POST_FORCE;
    mask |= POST_FORCE_RESPA;
    mask |= END_OF_STEP;
    mask |= THERMO_ENERGY;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixActivity::init()
{
    if (oflag && !atom->sphere_flag)
        error->all(FLERR,"Fix activity omega requires atom style sphere");
    if (ascale && !atom->ellipsoid_flag)
        error->all(FLERR,"Fix activity angmom requires atom style ellipsoid");
    
    // check variable
    
    if (tstr) {
        tvar = input->variable->find(tstr);
        if (tvar < 0)
            error->all(FLERR,"Variable name for fix activity does not exist");
        if (input->variable->equalstyle(tvar)) tstyle = EQUAL;
        else if (input->variable->atomstyle(tvar)) tstyle = ATOM;
        else error->all(FLERR,"Variable for fix activity is invalid style");
    }
    
    // if oflag or ascale set, check that all group particles are finite-size
    
    if (oflag) {
        double *radius = atom->radius;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit)
                if (radius[i] == 0.0)
                    error->one(FLERR,"Fix activity omega requires extended particles");
    }
    
    if (ascale) {
        avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
        if (!avec)
            error->all(FLERR,"Fix activity angmom requires atom style ellipsoid");
        
        int *ellipsoid = atom->ellipsoid;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit)
                if (ellipsoid[i] < 0)
                    error->one(FLERR,"Fix activity angmom requires extended particles");
    }
    
    // set force prefactors
    
    if (!atom->rmass) {
        for (int i = 1; i <= atom->ntypes; i++) {
            gfactor1[i] = -atom->mass[i] / t_period / force->ftm2v;
            gfactor2[i] = sqrt(atom->mass[i]) *
            sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
            force->ftm2v;
            gfactor1[i] *= 1.0/ratio[i];
            gfactor2[i] *= 1.0/sqrt(ratio[i]);
        }
    }
    
    if (temperature && temperature->tempbias) tbiasflag = BIAS;
    else tbiasflag = NOBIAS;
    
    if (strstr(update->integrate_style,"respa"))
        nlevels_respa = ((Respa *) update->integrate)->nlevels;
    
    if (gjfflag) gjffac = 1.0/(1.0+update->dt/2.0/t_period);
    
}

/* ---------------------------------------------------------------------- */

void FixActivity::setup(int vflag)
{
    if (strstr(update->integrate_style,"verlet"))
        post_force(vflag);
    else {
        ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
        post_force_respa(vflag,nlevels_respa-1,0);
        ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
    }
}

/* ---------------------------------------------------------------------- */

void FixActivity::post_force(int vflag)
{
    double *rmass = atom->rmass;
    
    // enumerate all 2^6 possibilities for template parameters
    // this avoids testing them inside inner loop:
    // TSTYLEATOM, GJF, TALLY, BIAS, RMASS, ZERO
    
#ifdef TEMPLATED_FIX_ACTIVITY
    if (tstyle == ATOM)
        if (gjfflag)
            if (tallyflag)
                if (tbiasflag == BIAS)
                    if (rmass)
                        if (zeroflag) post_force_templated<1,1,1,1,1,1>();
                        else          post_force_templated<1,1,1,1,1,0>();
                        else
                            if (zeroflag) post_force_templated<1,1,1,1,0,1>();
                            else          post_force_templated<1,1,1,1,0,0>();
                            else
                                if (rmass)
                                    if (zeroflag) post_force_templated<1,1,1,0,1,1>();
                                    else          post_force_templated<1,1,1,0,1,0>();
                                    else
                                        if (zeroflag) post_force_templated<1,1,1,0,0,1>();
                                        else          post_force_templated<1,1,1,0,0,0>();
                                        else
                                            if (tbiasflag == BIAS)
                                                if (rmass)
                                                    if (zeroflag) post_force_templated<1,1,0,1,1,1>();
                                                    else          post_force_templated<1,1,0,1,1,0>();
                                                    else
                                                        if (zeroflag) post_force_templated<1,1,0,1,0,1>();
                                                        else          post_force_templated<1,1,0,1,0,0>();
                                                        else
                                                            if (rmass)
                                                                if (zeroflag) post_force_templated<1,1,0,0,1,1>();
                                                                else          post_force_templated<1,1,0,0,1,0>();
                                                                else
                                                                    if (zeroflag) post_force_templated<1,1,0,0,0,1>();
                                                                    else          post_force_templated<1,1,0,0,0,0>();
                                                                    else
                                                                        if (tallyflag)
                                                                            if (tbiasflag == BIAS)
                                                                                if (rmass)
                                                                                    if (zeroflag) post_force_templated<1,0,1,1,1,1>();
                                                                                    else          post_force_templated<1,0,1,1,1,0>();
                                                                                    else
                                                                                        if (zeroflag) post_force_templated<1,0,1,1,0,1>();
                                                                                        else          post_force_templated<1,0,1,1,0,0>();
                                                                                        else
                                                                                            if (rmass)
                                                                                                if (zeroflag) post_force_templated<1,0,1,0,1,1>();
                                                                                                else          post_force_templated<1,0,1,0,1,0>();
                                                                                                else
                                                                                                    if (zeroflag) post_force_templated<1,0,1,0,0,1>();
                                                                                                    else          post_force_templated<1,0,1,0,0,0>();
                                                                                                    else
                                                                                                        if (tbiasflag == BIAS)
                                                                                                            if (rmass)
                                                                                                                if (zeroflag) post_force_templated<1,0,0,1,1,1>();
                                                                                                                else          post_force_templated<1,0,0,1,1,0>();
                                                                                                                else
                                                                                                                    if (zeroflag) post_force_templated<1,0,0,1,0,1>();
                                                                                                                    else          post_force_templated<1,0,0,1,0,0>();
                                                                                                                    else
                                                                                                                        if (rmass)
                                                                                                                            if (zeroflag) post_force_templated<1,0,0,0,1,1>();
                                                                                                                            else          post_force_templated<1,0,0,0,1,0>();
                                                                                                                            else
                                                                                                                                if (zeroflag) post_force_templated<1,0,0,0,0,1>();
                                                                                                                                else          post_force_templated<1,0,0,0,0,0>();
                                                                                                                                else
                                                                                                                                    if (gjfflag)
                                                                                                                                        if (tallyflag)
                                                                                                                                            if (tbiasflag == BIAS)
                                                                                                                                                if (rmass)
                                                                                                                                                    if (zeroflag) post_force_templated<0,1,1,1,1,1>();
                                                                                                                                                    else          post_force_templated<0,1,1,1,1,0>();
                                                                                                                                                    else
                                                                                                                                                        if (zeroflag) post_force_templated<0,1,1,1,0,1>();
                                                                                                                                                        else          post_force_templated<0,1,1,1,0,0>();
                                                                                                                                                        else
                                                                                                                                                            if (rmass)
                                                                                                                                                                if (zeroflag) post_force_templated<0,1,1,0,1,1>();
                                                                                                                                                                else          post_force_templated<0,1,1,0,1,0>();
                                                                                                                                                                else
                                                                                                                                                                    if (zeroflag) post_force_templated<0,1,1,0,0,1>();
                                                                                                                                                                    else          post_force_templated<0,1,1,0,0,0>();
                                                                                                                                                                    else
                                                                                                                                                                        if (tbiasflag == BIAS)
                                                                                                                                                                            if (rmass)
                                                                                                                                                                                if (zeroflag) post_force_templated<0,1,0,1,1,1>();
                                                                                                                                                                                else          post_force_templated<0,1,0,1,1,0>();
                                                                                                                                                                                else
                                                                                                                                                                                    if (zeroflag) post_force_templated<0,1,0,1,0,1>();
                                                                                                                                                                                    else          post_force_templated<0,1,0,1,0,0>();
                                                                                                                                                                                    else
                                                                                                                                                                                        if (rmass)
                                                                                                                                                                                            if (zeroflag) post_force_templated<0,1,0,0,1,1>();
                                                                                                                                                                                            else          post_force_templated<0,1,0,0,1,0>();
                                                                                                                                                                                            else
                                                                                                                                                                                                if (zeroflag) post_force_templated<0,1,0,0,0,1>();
                                                                                                                                                                                                else          post_force_templated<0,1,0,0,0,0>();
                                                                                                                                                                                                else
                                                                                                                                                                                                    if (tallyflag)
                                                                                                                                                                                                        if (tbiasflag == BIAS)
                                                                                                                                                                                                            if (rmass)
                                                                                                                                                                                                                if (zeroflag) post_force_templated<0,0,1,1,1,1>();
                                                                                                                                                                                                                else          post_force_templated<0,0,1,1,1,0>();
                                                                                                                                                                                                                else
                                                                                                                                                                                                                    if (zeroflag) post_force_templated<0,0,1,1,0,1>();
                                                                                                                                                                                                                    else          post_force_templated<0,0,1,1,0,0>();
                                                                                                                                                                                                                    else
                                                                                                                                                                                                                        if (rmass)
                                                                                                                                                                                                                            if (zeroflag) post_force_templated<0,0,1,0,1,1>();
                                                                                                                                                                                                                            else          post_force_templated<0,0,1,0,1,0>();
                                                                                                                                                                                                                            else
                                                                                                                                                                                                                                if (zeroflag) post_force_templated<0,0,1,0,0,1>();
                                                                                                                                                                                                                                else          post_force_templated<0,0,1,0,0,0>();
                                                                                                                                                                                                                                else
                                                                                                                                                                                                                                    if (tbiasflag == BIAS)
                                                                                                                                                                                                                                        if (rmass)
                                                                                                                                                                                                                                            if (zeroflag) post_force_templated<0,0,0,1,1,1>();
                                                                                                                                                                                                                                            else          post_force_templated<0,0,0,1,1,0>();
                                                                                                                                                                                                                                            else
                                                                                                                                                                                                                                                if (zeroflag) post_force_templated<0,0,0,1,0,1>();
                                                                                                                                                                                                                                                else          post_force_templated<0,0,0,1,0,0>();
                                                                                                                                                                                                                                                else
                                                                                                                                                                                                                                                    if (rmass)
                                                                                                                                                                                                                                                        if (zeroflag) post_force_templated<0,0,0,0,1,1>();
                                                                                                                                                                                                                                                        else          post_force_templated<0,0,0,0,1,0>();
                                                                                                                                                                                                                                                        else
                                                                                                                                                                                                                                                            if (zeroflag) post_force_templated<0,0,0,0,0,1>();
                                                                                                                                                                                                                                                            else          post_force_templated<0,0,0,0,0,0>();
#else
    post_force_untemplated(int(tstyle==ATOM), gjfflag, tallyflag,
                           int(tbiasflag==BIAS), int(rmass!=NULL), zeroflag);
#endif
}

/* ---------------------------------------------------------------------- */

void FixActivity::post_force_respa(int vflag, int ilevel, int iloop)
{
    if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ----------------------------------------------------------------------
 modify forces using one of the many activity styles
 ------------------------------------------------------------------------- */

#ifdef TEMPLATED_FIX_ACTIVITY
template < int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
int Tp_BIAS, int Tp_RMASS, int Tp_ZERO >
void FixActivity::post_force_templated()
#else
void FixActivity::post_force_untemplated
(int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
 int Tp_BIAS, int Tp_RMASS, int Tp_ZERO)
#endif
{
    double gamma1,gamma2;
    
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    // apply damping and thermostat to atoms in group
    
    // for Tp_TSTYLEATOM:
    //   use per-atom per-coord target temperature
    // for Tp_GJF:
    //   use Gronbech-Jensen/Farago algorithm
    //   else use regular algorithm
    // for Tp_TALLY:
    //   store drag plus random forces in factivity[nlocal][3]
    // for Tp_BIAS:
    //   calculate temperature since some computes require temp
    //   computed on current nlocal atoms to remove bias
    //   test v = 0 since some computes mask non-participating atoms via v = 0
    //   and added force has extra term not multiplied by v = 0
    // for Tp_RMASS:
    //   use per-atom masses
    //   else use per-type masses
    // for Tp_ZERO:
    //   sum random force over all atoms in group
    //   subtract sum/count from each atom in group
    
    double fdrag[3],fran[3],fsum[3],fsumall[3];
    bigint count;
    double fswap;
    
    double boltz = force->boltz;
    double dt = update->dt;
    double mvv2e = force->mvv2e;
    double ftm2v = force->ftm2v;
    
    compute_target();
    
    if (Tp_ZERO) {
        fsum[0] = fsum[1] = fsum[2] = 0.0;
        count = group->count(igroup);
        if (count == 0)
            error->all(FLERR,"Cannot zero activity force of 0 atoms");
    }
    
    // reallocate factivity if necessary
    
    if (Tp_TALLY) {
        if (atom->nmax > maxatom1) {
            memory->destroy(factivity);
            maxatom1 = atom->nmax;
            memory->create(factivity,maxatom1,3,"activity:factivity");
        }
        factivity_allocated = 1;
    }
    
    if (Tp_BIAS) temperature->compute_scalar();
    
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            if (Tp_TSTYLEATOM) tsqrt = sqrt(tforce[i]);
            if (Tp_RMASS) {
                // YOU EXECUTE THIS IF STATMENT
                // We're going to replace this with the same format as fix_viscous
                gamma2 = sqrt( 2 * boltz * t_period ) * tsqrt;
                gamma1 = -t_period;
                // Essentially, this is controlled by values of T and dt (only values that are != 1)
                /*
                gamma1 = -rmass[i] / t_period / ftm2v;
                gamma2 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
                gamma1 *= 1.0/ratio[type[i]];
                gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
                */
                /*
                std::cout << "\n" << "sqrt(rmass[i]) = " << sqrt(rmass[i]) <<
                ", boltz = " << boltz <<
                ", t_period = " << t_period <<
                ", dt = " << dt <<
                ", mvv2e = " << mvv2e <<
                ", ftm2v = " << ftm2v <<
                ", ratio[type[i]] = " << ratio[type[i]] <<
                ", tsqrt = " << tsqrt << "\n";
                */
            } else {
                //gamma1 = gfactor1[type[i]];
                //gamma2 = gfactor2[type[i]] * tsqrt;
            }
            
            fran[0] = gamma2*(random->uniform()-0.5);
            fran[1] = gamma2*(random->uniform()-0.5);
            fran[2] = 0;//gamma2*(random->uniform()-0.5);
            
            if (Tp_BIAS) {
                temperature->remove_bias(i,v[i]);
                fdrag[0] = gamma1*v[i][0];
                fdrag[1] = gamma1*v[i][1];
                fdrag[2] = gamma1*v[i][2];
                if (v[i][0] == 0.0) fran[0] = 0.0;
                if (v[i][1] == 0.0) fran[1] = 0.0;
                if (v[i][2] == 0.0) fran[2] = 0.0;
                temperature->restore_bias(i,v[i]);
            } else {
                fdrag[0] = gamma1*v[i][0];
                fdrag[1] = gamma1*v[i][1];
                fdrag[2] = 0;//gamma1*v[i][2];
            }
            
            if (Tp_GJF) {
                fswap = 0.5*(fran[0]+franprev[i][0]);
                franprev[i][0] = fran[0];
                fran[0] = fswap;
                fswap = 0.5*(fran[1]+franprev[i][1]);
                franprev[i][1] = fran[1];
                fran[1] = fswap;
                fswap = 0.5*(fran[2]+franprev[i][2]);
                franprev[i][2] = fran[2];
                fran[2] = fswap;
                
                fdrag[0] *= gjffac;
                fdrag[1] *= gjffac;
                fdrag[2] *= gjffac;
                fran[0] *= gjffac;
                fran[1] *= gjffac;
                fran[2] *= gjffac;
                f[i][0] *= gjffac;
                f[i][1] *= gjffac;
                f[i][2] *= gjffac;
            }
            
            f[i][0] += fdrag[0] + fran[0];
            f[i][1] += fdrag[1] + fran[1];
            f[i][2] += 0;//fdrag[2] + fran[2];
            
            if (Tp_TALLY) {
                factivity[i][0] = fdrag[0] + fran[0];
                factivity[i][1] = fdrag[1] + fran[1];
                factivity[i][2] = fdrag[2] + fran[2];
            }
            
            if (Tp_ZERO) {
                fsum[0] += fran[0];
                fsum[1] += fran[1];
                fsum[2] += fran[2];
            }
        }
    }
    
    // set total force to zero
    
    if (Tp_ZERO) {
        MPI_Allreduce(fsum,fsumall,3,MPI_DOUBLE,MPI_SUM,world);
        fsumall[0] /= count;
        fsumall[1] /= count;
        fsumall[2] /= count;
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                f[i][0] -= fsumall[0];
                f[i][1] -= fsumall[1];
                f[i][2] -= fsumall[2];
            }
        }
    }
    
    // thermostat omega and angmom
    
    determine_activity();
    //std::cout << "\n" << "f[1][0] = " << f[1][0] << ", f[1][1] = " << f[1][1] << "\n";
    if (oflag) omega_thermostat();
    if (ascale) angmom_thermostat();
}

/* ----------------------------------------------------------------------
 compute the active component of f[i]
 ------------------------------------------------------------------------- */
void FixActivity::determine_activity()
{
    double vprop;
    double boltz = force->boltz;
    double dt = update->dt;
    double mvv2e = force->mvv2e;
    double ftm2v = force->ftm2v;
    double factive[3];
    
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    double **f = atom->f;
    bigint count;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double *shape,*myquat;
    double gamma3;
    
    
    
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            
            shape = bonus[ellipsoid[i]].shape;
            // definition of vprop (from struct and dyn of phase sep active colloidal fluid)
            vprop = peclet * tsqrt * tsqrt * boltz / shape[0] / t_period;
            // effectively this results in F_act = Pe*kT/sigma 
            gamma3 = t_period;
            myquat = bonus[ellipsoid[i]].quat;
            /*
            if (i == 0)
            {
                std::cout << "The QUATERNION of the " << i << "th particle is, myquat[0] = "
                << myquat[0] << " , x mult factor = " << (cos(2*(acos(myquat[0])))) << " and y mult factor = "
                << (sin(2*(acos(myquat[0])))) << "\n";
            }
            */
            // F_active is essentially the peclet number times the temperature (making it constant, this fixes T)
            factive[0] = (cos(2*(acos(myquat[0])))) * gamma3 * vprop;
            factive[1] = (sin(2*(acos(myquat[0])))) * gamma3 * vprop;
            factive[2] = 0;
     
            f[i][0] += factive[0];
            f[i][1] += factive[1];
            f[i][2] += factive[2];
            /*
            if (i >= 0 && i <= 4)
            {
                std::cout << "------" << i << "------" << "\n";
                std::cout << "X ACTIVITY = " << factive[0] << ", Y ACTIVITY = " << factive[1] << "\n";
                std::cout << "X = " << f[i][0] << ", Y = " << f[i][1] << ", Z = " << f[i][2] << "\n";
                std::cout << "The Y COMPONENT is " << f[i][1]/f[i][0]*100 << "% of the x component.\n";
            }
            */
        }
    }
}

/* ----------------------------------------------------------------------
 set current t_target and t_sqrt
 ------------------------------------------------------------------------- */

void FixActivity::compute_target()
{
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    double delta = update->ntimestep - update->beginstep;
    if (delta != 0.0) delta /= update->endstep - update->beginstep;
    
    // if variable temp, evaluate variable, wrap with clear/add
    // reallocate tforce array if necessary
    
    if (tstyle == CONSTANT) {
        t_target = t_start + delta * (t_stop-t_start);
        tsqrt = sqrt(t_target);
    } else {
        modify->clearstep_compute();
        if (tstyle == EQUAL) {
            t_target = input->variable->compute_equal(tvar);
            if (t_target < 0.0)
                error->one(FLERR,"Fix activity variable returned negative temperature");
            tsqrt = sqrt(t_target);
        } else {
            if (atom->nmax > maxatom2) {
                maxatom2 = atom->nmax;
                memory->destroy(tforce);
                memory->create(tforce,maxatom2,"activity:tforce");
            }
            input->variable->compute_atom(tvar,igroup,tforce,1,0);
            for (int i = 0; i < nlocal; i++)
                if (mask[i] & groupbit)
                    if (tforce[i] < 0.0)
                        error->one(FLERR,
                                   "Fix activity variable returned negative temperature");
        }
        modify->addstep_compute(update->ntimestep + 1);
    }
}

/* ----------------------------------------------------------------------
 thermostat rotational dof via omega
 ------------------------------------------------------------------------- */

void FixActivity::omega_thermostat()
{
    double gamma1,gamma2;
    
    double boltz = force->boltz;
    double dt = update->dt;
    double mvv2e = force->mvv2e;
    double ftm2v = force->ftm2v;
    
    double **torque = atom->torque;
    double **omega = atom->omega;
    double *radius = atom->radius;
    double *rmass = atom->rmass;
    int *mask = atom->mask;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    
    // rescale gamma1/gamma2 by 10/3 & sqrt(10/3) for spherical particles
    // does not affect rotational thermosatting
    // gives correct rotational diffusivity behavior
    
    double tendivthree = 10.0/3.0;
    double tran[3];
    double inertiaone;
    
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (radius[i] > 0.0)) {
            inertiaone = SINERTIA*radius[i]*radius[i]*rmass[i];
            if (tstyle == ATOM) tsqrt = sqrt(tforce[i]);
            gamma1 = -tendivthree*inertiaone / t_period / ftm2v;
            gamma2 = sqrt(inertiaone) * sqrt(80.0*boltz/t_period/dt/mvv2e) / ftm2v;
            gamma1 *= 1.0/ratio[type[i]];
            gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
            tran[0] = gamma2*(random->uniform()-0.5);
            tran[1] = gamma2*(random->uniform()-0.5);
            tran[2] = gamma2*(random->uniform()-0.5);
            torque[i][0] += gamma1*omega[i][0] + tran[0];
            torque[i][1] += gamma1*omega[i][1] + tran[1];
            torque[i][2] += gamma1*omega[i][2] + tran[2];
        }
    }
}

/* ----------------------------------------------------------------------
 thermostat rotational dof via angmom
 ------------------------------------------------------------------------- */

void FixActivity::angmom_thermostat()
{
    double gamma1,gamma2;
    
    double boltz = force->boltz;
    double dt = update->dt;
    double mvv2e = force->mvv2e;
    double ftm2v = force->ftm2v;
    
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    double **torque = atom->torque;
    double **omega = atom->omega;
    double **angmom = atom->angmom;
    double *rmass = atom->rmass;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    
    // rescale gamma1/gamma2 by ascale for aspherical particles
    // does not affect rotational thermosatting
    // gives correct rotational diffusivity behavior if (nearly) spherical
    // any value will be incorrect for rotational diffusivity if aspherical
    
    double inertia[3],tran[3];
    double *shape,*quat;
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            shape = bonus[ellipsoid[i]].shape;
            // We actually want SINERTIA not EINERTIA
            inertia[0] = SINERTIA*rmass[i] * (shape[1]*shape[1]+shape[2]*shape[2]);
            inertia[1] = SINERTIA*rmass[i] * (shape[0]*shape[0]+shape[2]*shape[2]);
            inertia[2] = SINERTIA*rmass[i] * (shape[0]*shape[0]+shape[1]*shape[1]);
            quat = bonus[ellipsoid[i]].quat;
            // This computes omega... we want our own omega
            //MathExtra::mq_to_omega(angmom[i],quat,inertia,omega);
            
            if (tstyle == ATOM) tsqrt = sqrt(tforce[i]);
            /* OLD WAY
            //gamma1 = -ascale / t_period / ftm2v;
            //gamma2 = sqrt(ascale*24.0*boltz/t_period/dt/mvv2e) / ftm2v;
            //gamma1 *= 1.0/ratio[type[i]];
            //gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
            //tran[0] = sqrt(inertia[0])*gamma2*(random->uniform()-0.5);
            //tran[1] = sqrt(inertia[1])*gamma2*(random->uniform()-0.5);
            //tran[2] = sqrt(inertia[2])*gamma2*(random->uniform()-0.5);
            */
            
            // angular momentum = angular velocity x inertia
            // paper gives, omega = rad(2Dr) x rndm_numbr
            
            /*  OLD WAY
            torque[i][0] += inertia[0]*gamma1*omega[0] + tran[0];
            torque[i][1] += inertia[1]*gamma1*omega[1] + tran[1];
            torque[i][2] += inertia[2]*gamma1*omega[2] + tran[2];
            */
            // IMPLEMENT OMEGA AS YOU WOULD A SPHERE (multiply by tran instead?)
            // make gamma relative to random translational force
            
            // gamma1 is the coefficient to the random value == rad(2Dr)
            //gamma1 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
            //gamma1 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
            //gamma1 *= sqrt(3);
            // Use fix_viscous
            // Ideally this contains a sigma^2 in the denominator
            gamma1 = sqrt( 6 * boltz / t_period / shape[0] / shape[0] ) * tsqrt;
            // Do I need a gamma factor in front of omega?
            //gamma2 = -tendivthree*inertiaone / t_period / ftm2v;
            
            tran[0] = gamma1 * (random->uniform()-0.5);
            tran[1] = gamma1 * (random->uniform()-0.5);
            tran[2] = gamma1 * (random->uniform()-0.5);
            
            // gamma1 is the coefficient of the RANDOM TERM
            
            //omega[0] = gamma1*tran[0];
            //omega[1] = gamma1*tran[1];
            
            // There can only be torque, angular velocity, angmom from the z-component
            //torque[i][0] += inertia[0]*omega[0] + tran[0];
            //torque[i][1] += inertia[1]*omega[1] + tran[1];
            torque[i][0] += 0;
            torque[i][1] += 0;
            torque[i][2] += tran[2] - omega[i][2];
        }
    }
}

/* ----------------------------------------------------------------------
 tally energy transfer to thermal reservoir
 ------------------------------------------------------------------------- */

void FixActivity::end_of_step()
{
    if (!tallyflag) return;
    
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    energy_onestep = 0.0;
    
    for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
            energy_onestep += factivity[i][0]*v[i][0] + factivity[i][1]*v[i][1] +
            factivity[i][2]*v[i][2];
    
    energy += energy_onestep*update->dt;
}

/* ---------------------------------------------------------------------- */

void FixActivity::reset_target(double t_new)
{
    t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixActivity::reset_dt()
{
    if (atom->mass) {
        for (int i = 1; i <= atom->ntypes; i++) {
            gfactor2[i] = sqrt(atom->mass[i]) *
            sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
            force->ftm2v;
            gfactor2[i] *= 1.0/sqrt(ratio[i]);
        }
    }
}

/* ---------------------------------------------------------------------- */

int FixActivity::modify_param(int narg, char **arg)
{
    if (strcmp(arg[0],"temp") == 0) {
        if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
        delete [] id_temp;
        int n = strlen(arg[1]) + 1;
        id_temp = new char[n];
        strcpy(id_temp,arg[1]);
        
        int icompute = modify->find_compute(id_temp);
        if (icompute < 0)
            error->all(FLERR,"Could not find fix_modify temperature ID");
        temperature = modify->compute[icompute];
        
        if (temperature->tempflag == 0)
            error->all(FLERR,
                       "Fix_modify temperature ID does not compute temperature");
        if (temperature->igroup != igroup && comm->me == 0)
            error->warning(FLERR,"Group for fix_modify temp != fix group");
        return 2;
    }
    return 0;
}

/* ---------------------------------------------------------------------- */

double FixActivity::compute_scalar()
{
    if (!tallyflag || !factivity_allocated) return 0.0;
    
    // capture the very first energy transfer to thermal reservoir
    
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    if (update->ntimestep == update->beginstep) {
        energy_onestep = 0.0;
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit)
                energy_onestep += factivity[i][0]*v[i][0] + factivity[i][1]*v[i][1] +
                factivity[i][2]*v[i][2];
        energy = 0.5*energy_onestep*update->dt;
    }
    
    // convert midstep energy back to previous fullstep energy
    
    double energy_me = energy - 0.5*energy_onestep*update->dt;
    
    double energy_all;
    MPI_Allreduce(&energy_me,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    return -energy_all;
}

/* ----------------------------------------------------------------------
 extract thermostat properties
 ------------------------------------------------------------------------- */

void *FixActivity::extract(const char *str, int &dim)
{
    dim = 0;
    if (strcmp(str,"t_target") == 0) {
        return &t_target;
    }
    return NULL;
}

/* ----------------------------------------------------------------------
 memory usage of tally array
 ------------------------------------------------------------------------- */

double FixActivity::memory_usage()
{
    double bytes = 0.0;
    if (gjfflag) bytes += atom->nmax*3 * sizeof(double);
    if (tallyflag) bytes += atom->nmax*3 * sizeof(double);
    if (tforce) bytes += atom->nmax * sizeof(double);
    return bytes;
}

/* ----------------------------------------------------------------------
 allocate atom-based array for franprev
 ------------------------------------------------------------------------- */

void FixActivity::grow_arrays(int nmax)
{
    memory->grow(franprev,nmax,3,"fix_activity:franprev");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based array
 ------------------------------------------------------------------------- */

void FixActivity::copy_arrays(int i, int j, int delflag)
{
    for (int m = 0; m < nvalues; m++)
        franprev[j][m] = franprev[i][m];
}

/* ----------------------------------------------------------------------
 pack values in local atom-based array for exchange with another proc
 ------------------------------------------------------------------------- */

int FixActivity::pack_exchange(int i, double *buf)
{
    for (int m = 0; m < nvalues; m++) buf[m] = franprev[i][m];
    return nvalues;
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based array from exchange with another proc
 ------------------------------------------------------------------------- */

int FixActivity::unpack_exchange(int nlocal, double *buf)
{
    for (int m = 0; m < nvalues; m++) franprev[nlocal][m] = buf[m];
    return nvalues;
}
