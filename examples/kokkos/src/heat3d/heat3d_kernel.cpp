/**
 * A Head3D benchmark modified from 
 * https://github.com/kokkos/kokkos-remote-spaces/tree/main/applications/heat3d
 * 
 * It will use VLCs in place of MPI
 *  
*/

#include <Kokkos_Core.hpp>
#include <typeinfo>
#include <iostream>

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create(int me) { return ExecSpace(); }
  static void destroy(ExecSpace&) {}
  static bool overlap() { return false; }
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create(int me) {
    cudaStream_t stream;
    cudaSetDevice(me);  // me equals to device id
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(int me, Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaSetDevice(me);
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value          = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif

struct CommHelper {
  // MPI_Comm comm;

  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // My rank
  int me;

  // N ranks
  int nranks;

  // My pos in proc grid
  int x, y, z;

  // Neighbor Ranks
  int up, down, left, right, front, back;

  CommHelper(int me_, int nranks_) {
    // comm = comm_;

    // MPI_Comm_size(comm, &nranks);
    // MPI_Comm_rank(comm, &me);
    me = me_;
    nranks = nranks_;

    nx = std::pow(1.0 * nranks, 1.0 / 3.0);
    while (nranks % nx != 0) nx++;

    ny = std::sqrt(1.0 * (nranks / nx));
    while ((nranks / nx) % ny != 0) ny++;

    nz    = nranks / nx / ny;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = x == 0 ? -1 : me - 1;
    right = x == nx - 1 ? -1 : me + 1;
    down  = y == 0 ? -1 : me - nx;
    up    = y == ny - 1 ? -1 : me + nx;
    front = z == 0 ? -1 : me - nx * ny;
    back  = z == nz - 1 ? -1 : me + nx * ny;
  }

  // template <class ViewType>
  // void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer,
  //                  MPI_Request* request_send, MPI_Request* request_recv) {
  //   MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1,
  //             comm, request_recv);
  //   MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1,
  //             comm, request_send);
  // }
};

struct System {
  // Using theoretical physicists way of describing system,
  // i.e. we stick everything in as few constants as possible
  // be i and i+1 two timesteps dt apart:
  // T(x,y,z)_(i+1) = T(x,y,z)_(i)+dT(x,y,z)*dt;
  // dT(x,y,z) = q * sum_dxdydz( T(x+dx,y+dy,z+dz) - T(x,y,z) )
  // If its surface of the body add:
  // dT(x,y,z) += -sigma*T(x,y,z)^4
  // If its z==0 surface add incoming radiation energy
  // dT(x,y,0) += P

  // Communicator
  CommHelper comm;
  // MPI_Request mpi_requests_recv[6];
  // MPI_Request mpi_requests_send[6];
  // int mpi_active_requests;

  // size of system
  int X, Y, Z;

  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  // number of timesteps
  int N;

  // interval for print
  int I;

  // Temperature and delta Temperature
  Kokkos::View<double***> T, dT;
  // Halo data
  using buffer_t =
      Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  buffer_t T_left, T_right, T_up, T_down, T_front, T_back;
  buffer_t T_left_out, T_right_out, T_up_out, T_down_out, T_front_out,
      T_back_out;

  Kokkos::DefaultExecutionSpace E_left, E_right, E_up, E_down, E_front, E_back,
      E_bulk;

  // Initial temperature
  double T0;

  // timestep width
  double dt;

  // thermal transfer coefficient
  double q;

  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double sigma;

  // incoming power
  double P;

  // init_system

  System(int me, int nranks) : comm(me, nranks) {
    // mpi_active_requests = 0;
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X;
    N                  = 10000;
    I = N - 1;
    T       = Kokkos::View<double***>();
    dT      = Kokkos::View<double***>();
    T0      = 0.0;
    dt      = 0.1;
    q       = 1.0;
    sigma   = 1.0;
    P       = 1.0;
    E_left  = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_right = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_up    = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_down  = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_front = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_back  = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
    E_bulk  = SpaceInstance<Kokkos::DefaultExecutionSpace>::create(me);
  }

  void destroy_exec_spaces() {
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_left);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_right);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_front);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_back);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_up);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_down);
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(comm.me, E_bulk);
  }

  void setup_subdomain() {
    int dX = (X + comm.nx - 1) / comm.nx;
    X_lo   = dX * comm.x;
    X_hi   = X_lo + dX;
    if (X_hi > X) X_hi = X;
    int dY = (Y + comm.ny - 1) / comm.ny;
    Y_lo   = dY * comm.y;
    Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z;
    Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;

    T  = Kokkos::View<double***>("System::T", X_hi - X_lo, Y_hi - Y_lo,
                                Z_hi - Z_lo);
    dT = Kokkos::View<double***>("System::dT", T.extent(0), T.extent(1),
                                 T.extent(2));
    Kokkos::deep_copy(T, T0);

    // incoming halos
    if (X_lo != 0)
      T_left = buffer_t("System::T_left", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X)
      T_right = buffer_t("System::T_right", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0)
      T_down = buffer_t("System::T_down", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y) T_up = buffer_t("System::T_up", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0)
      T_front = buffer_t("System::T_front", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z)
      T_back = buffer_t("System::T_back", X_hi - X_lo, Y_hi - Y_lo);

    // outgoing halo
    if (X_lo != 0)
      T_left_out = buffer_t("System::T_left_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X)
      T_right_out = buffer_t("System::T_right_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0)
      T_down_out = buffer_t("System::T_down_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y)
      T_up_out = buffer_t("System::T_up_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0)
      T_front_out = buffer_t("System::T_front_out", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z)
      T_back_out = buffer_t("System::T_back_out", X_hi - X_lo, Y_hi - Y_lo);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in X direction\n", X);
    printf("  -Y IARG: (%i) num elements in Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", N);
    printf("  -I IARG: (%i) print interval\n", I);
    printf("  -T0 FARG: (%lf) initial temperature\n", T0);
    printf("  -dt FARG: (%lf) timestep size\n", dt);
    printf("  -q FARG: (%lf) thermal conductivity\n", q);
    printf("  -sigma FARG: (%lf) thermal radiation\n", sigma);
    printf("  -P FARG: (%lf) incoming power\n", P);
  }

  // check command line args
  void check_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        print_help();
        exit(0);
      }
    }
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-X") == 0) X = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Y") == 0) Y = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Z") == 0) Z = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-I") == 0) I = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-T0") == 0) T0 = atof(argv[i + 1]);
      if (strcmp(argv[i], "-dt") == 0) dt = atof(argv[i + 1]);
      if (strcmp(argv[i], "-q") == 0) q = atof(argv[i + 1]);
      if (strcmp(argv[i], "-sigma") == 0) sigma = atof(argv[i + 1]);
      if (strcmp(argv[i], "-P") == 0) P = atof(argv[i + 1]);
    }
    setup_subdomain();
  }

  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    dT(x, y, z) = dT_xyz;
  }
  void compute_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T.extent(0);
    int myY = T.extent(1);
    int myZ = T.extent(2);
    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  };

  // Compute non-exposed surface
  // Dispatch makes sure that we don't hit elements twice
  enum { left, right, down, up, front, back };

  template <int Surface>
  struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i,
                                  int j) const {
    int NX = T.extent(0);
    int NY = T.extent(1);
    int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) {
      x = 0;
      y = i;
      z = j;
    }
    if (Surface == right) {
      x = NX - 1;
      y = i;
      z = j;
    }
    if (Surface == down) {
      x = i;
      y = 0;
      z = j;
    }
    if (Surface == up) {
      x = i;
      y = NY - 1;
      z = j;
    }
    if (Surface == front) {
      x = i;
      y = j;
      z = 0;
    }
    if (Surface == back) {
      x = i;
      y = j;
      z = NZ - 1;
    }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    // Heat conduction to inner body
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0) dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X) dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0) dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y) dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0) dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z) dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void pack_T_halo() {
    // mpi_active_requests = 0;
    // int mar             = 0;
    if (X_lo != 0) {
      Kokkos::deep_copy(E_left, T_left_out,
                        Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL));
      // mar++;
    }
    if (Y_lo != 0) {
      Kokkos::deep_copy(E_down, T_down_out,
                        Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL));
    }
    if (Z_lo != 0) {
      Kokkos::deep_copy(E_front, T_front_out,
                        Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0));
    }
    if (X_hi != X) {
      Kokkos::deep_copy(
          E_right, T_right_out,
          Kokkos::subview(T, X_hi - X_lo - 1, Kokkos::ALL, Kokkos::ALL));
    }
    if (Y_hi != Y) {
      Kokkos::deep_copy(
          E_up, T_up_out,
          Kokkos::subview(T, Kokkos::ALL, Y_hi - Y_lo - 1, Kokkos::ALL));
    }
    if (Z_hi != Z) {
      Kokkos::deep_copy(
          E_back, T_back_out,
          Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, Z_hi - Z_lo - 1));
    }
  }

  void exchange_T_halo(System *other_sys) {
    if (X_lo != 0) {
      E_left.fence();
      std::cout << typeid(other_sys->T_left_out.data()).name() << std::endl;
      Kokkos::deep_copy(E_left, Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL), other_sys->T_left_out);
      // comm.isend_irecv(comm.left, T_left_out, T_left,
      //                  &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      // mar++;
    }
    if (Y_lo != 0) {
      E_down.fence();
      std::cout << "2: " << comm.me << " to " << comm.down << std::endl;
      Kokkos::deep_copy(E_down, Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL), other_sys->T_down_out);
    }
    if (Z_lo != 0) {
      // Kokkos::deep_copy(other_sys->T_back_out, T_front);
      cudaMemcpy(T_front.data(), other_sys->T_back_out.data(), (T_front.size()) * sizeof(double), cudaMemcpyDefault);
    }
    if (X_hi != X) {
      E_right.fence();
      std::cout << "4" << std::endl;
      Kokkos::deep_copy(E_right, Kokkos::subview(T, X_hi - X_lo - 1, Kokkos::ALL, Kokkos::ALL), other_sys->T_right_out);
    }
    if (Y_hi != Y) {
      E_up.fence();
      std::cout << "5" << std::endl;
      Kokkos::deep_copy(E_up, Kokkos::subview(T, Kokkos::ALL, Y_hi - Y_lo - 1, Kokkos::ALL), other_sys->T_up_out);
    }
    if (Z_hi != Z) {
      // Kokkos::deep_copy(other_sys->T_front_out, T_back);
      cudaMemcpy(T_back.data(), other_sys->T_front_out.data(), (T_back.size()) * sizeof(double), cudaMemcpyDefault);
    }
    // mpi_active_requests = mar;
  }

  void compute_surface_dT() {
    using policy_left_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;

    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    // if (mpi_active_requests > 0) {
    //   MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
    //   MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);
    // }
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Left",
        Kokkos::Experimental::require(
            policy_left_t(E_left, {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Right",
        Kokkos::Experimental::require(
            policy_right_t(E_right, {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Down",
        Kokkos::Experimental::require(
            policy_down_t(E_down, {1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Up",
        Kokkos::Experimental::require(
            policy_up_t(E_up, {1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_front",
        Kokkos::Experimental::require(
            policy_front_t(E_front, {1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_back",
        Kokkos::Experimental::require(
            policy_back_t(E_back, {1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  }

  // Some compilers have deduction issues if this were just a tagged operator
  // So did a full Functor here instead
  struct UpdateT {
    Kokkos::View<double***> T, dT;
    double dt;
    UpdateT(Kokkos::View<double***> T_, Kokkos::View<double***> dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int X       = T.extent(0);
    int Y       = T.extent(1);
    int Z       = T.extent(2);
    double my_T = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {0, 0, 0}, {X, Y, Z}, {10, 10, 10}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        UpdateT(T, dT, dt), my_T);
    // double sum_T;
    // MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return my_T;
  }
};

void *initialize_system(int me, int nranks, int argc, char* argv[]) {
  System *system = new System(me, nranks);
  system->check_args(argc, argv);
  return static_cast<void *>(system);
}

void finalize_system(void *sys) {
  System *system = static_cast<System *>(sys);
  system->destroy_exec_spaces();
}

int get_N(void *sys) {
  System *system = static_cast<System *>(sys);
  return system->N;
}

int get_I(void *sys) {
  System *system = static_cast<System *>(sys);
  return system->I;
}

void heat3d_phase1(void *sys, int t) {
  System *system = static_cast<System *>(sys);
  if (t > system->N / 2) system->P = 0.0;
  system->pack_T_halo();       // Overlap O1
  Kokkos::fence();
  std::cout << "pack T halo done" << std::endl;
  system->compute_inner_dT();  // Overlap O1
  Kokkos::fence();
  std::cout << "compute inner dT done" << std::endl;
}

void heat3d_exchange_data(void *this_sys, void *other_sys) {
  System *this_system = static_cast<System *>(this_sys);
  System *other_system = static_cast<System *>(other_sys);
  this_system->exchange_T_halo(other_system);
}

// return local T value
double heat3d_phase2(void *sys) {
  System *system = static_cast<System *>(sys);
  system->compute_surface_dT();
  Kokkos::fence();
  auto T = system->update_T();
  return T / (1e-9 * (system->X * system->Y * system->Z));
}

void kokkos_initialize(int dev) {
	Kokkos::initialize(Kokkos::InitializationSettings()  // (since 3.7)
                      .set_disable_warnings(false)
					            .set_device_id(dev));
}

void kokkos_finalize() {
	Kokkos::finalize();
}
