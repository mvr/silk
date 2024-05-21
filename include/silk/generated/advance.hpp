_DI_ bool inplace_advance_unknown(
        uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
        uint32_t &not_low, uint32_t &not_high, uint32_t &not_stable,
        uint32_t stator, int max_width = 28, int max_height = 28, uint32_t max_pop = 784, uint32_t* smem = nullptr
    ) {

    // obtain lower and upper bounds in binary:
    uint4 lb_bin = kc::sum16<false>(not_low, not_low & not_stable);
    uint4 ub_bin = kc::sum16<true>(not_high, not_high & not_stable);

    // lb[i] := "we know that (curr_pop - stable_pop >= i - 4)"
    uint32_t lb0 = lb_bin.w | lb_bin.z;
    uint32_t lb4 = lb_bin.w;
    uint32_t lb8 = lb_bin.w & lb_bin.z;
    uint32_t lb2 = lb4 | (lb0 & lb_bin.y);
    uint32_t lb6 = lb8 | (lb4 & lb_bin.y);
    uint32_t lb1 = lb2 | (lb0 & lb_bin.x);
    uint32_t lb3 = lb4 | (lb2 & lb_bin.x);
    uint32_t lb5 = lb6 | (lb4 & lb_bin.x);
    uint32_t lb7 = lb8 | (lb6 & lb_bin.x);

    // ub[i] := "we know that (curr_pop - stable_pop  < i - 4)"
    uint32_t ub8 = ub_bin.w | ub_bin.z;
    uint32_t ub4 = ub_bin.w;
    uint32_t ub0 = ub_bin.w & ub_bin.z;
    uint32_t ub6 = ub4 | (ub8 & ub_bin.y);
    uint32_t ub2 = ub0 | (ub4 & ub_bin.y);
    uint32_t ub7 = ub6 | (ub8 & ub_bin.x);
    uint32_t ub5 = ub4 | (ub6 & ub_bin.x);
    uint32_t ub3 = ub2 | (ub4 & ub_bin.x);
    uint32_t ub1 = ub0 | (ub2 & ub_bin.x);

    uint32_t gnot_low    = 0xffffffffu;
    uint32_t gnot_high   = 0xffffffffu;
    uint32_t gnot_stable = 0xffffffffu;
    uint32_t unstable_d0, unstable_d1, unstable_d2, unstable_l2, unstable_l3, unstable_d4, unstable_d5, unstable_d6;

    {
        uint32_t fd_next = (ub6 | (ub7 & not_high) | lb8) | ad0;
        uint32_t fl_next = (lb6 & (lb7 | not_stable) & ub8) | ad0;
        unstable_d0 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    {
        uint32_t fd_next = (ub5 | (ub6 & not_high) | lb7) | ad1;
        uint32_t fl_next = (lb5 & (lb6 | not_stable) & ub7) | ad1;
        unstable_d1 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    {
        uint32_t fd_next = (ub4 | (ub5 & not_high) | lb6) | ad2;
        uint32_t fl_next = (lb4 & (lb5 | not_stable) & ub6) | ad2;
        unstable_d2 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    {
        uint32_t fd_next = (ub4 | (ub5 & not_stable) | lb6) | al2;
        uint32_t fl_next = (lb4 & (lb5 | not_low) & ub6) | al2;
        unstable_l2 = fd_next;
        gnot_stable &= fd_next;
        gnot_low &= fl_next;
    }

    {
        uint32_t fd_next = (ub3 | (ub4 & not_stable) | lb5) | al3;
        uint32_t fl_next = (lb3 & (lb4 | not_low) & ub5) | al3;
        unstable_l3 = fd_next;
        gnot_stable &= fd_next;
        gnot_low &= fl_next;
    }

    {
        uint32_t fd_next = (ub2 | (ub3 & not_high) | lb4) | ad4;
        uint32_t fl_next = (lb2 & (lb3 | not_stable) & ub4) | ad4;
        unstable_d4 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    {
        uint32_t fd_next = (ub1 | (ub2 & not_high) | lb3) | ad5;
        uint32_t fl_next = (lb1 & (lb2 | not_stable) & ub3) | ad5;
        unstable_d5 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    {
        uint32_t fd_next = (ub0 | (ub1 & not_high) | lb2) | ad6;
        uint32_t fl_next = (lb0 & (lb1 | not_stable) & ub2) | ad6;
        unstable_d6 = fl_next;
        gnot_high &= fd_next;
        gnot_stable &= fl_next;
    }

    uint32_t forced_stable = kc::get_forced_stable(gnot_stable, ad0, stator, max_width, max_height, max_pop);
    gnot_low |= forced_stable;
    gnot_high |= forced_stable;
    uint32_t improvements = 0;
    uint32_t gad0 = ad0 | (forced_stable & unstable_d0); improvements |= (gad0 &~ ad0); ad0 = gad0;
    uint32_t gad1 = ad1 | (forced_stable & unstable_d1); improvements |= (gad1 &~ ad1); ad1 = gad1;
    uint32_t gad2 = ad2 | (forced_stable & unstable_d2); improvements |= (gad2 &~ ad2); ad2 = gad2;
    uint32_t gal2 = al2 | (forced_stable & unstable_l2); improvements |= (gal2 &~ al2); al2 = gal2;
    uint32_t gal3 = al3 | (forced_stable & unstable_l3); improvements |= (gal3 &~ al3); al3 = gal3;
    uint32_t gad4 = ad4 | (forced_stable & unstable_d4); improvements |= (gad4 &~ ad4); ad4 = gad4;
    uint32_t gad5 = ad5 | (forced_stable & unstable_d5); improvements |= (gad5 &~ ad5); ad5 = gad5;
    uint32_t gad6 = ad6 | (forced_stable & unstable_d6); improvements |= (gad6 &~ ad6); ad6 = gad6;

    // store in smem if appropriate:
    if (smem != nullptr) {
        smem[threadIdx.x] = unstable_d0;
        smem[threadIdx.x + 32] = unstable_d1;
        smem[threadIdx.x + 64] = unstable_d2;
        smem[threadIdx.x + 96] = unstable_l2;
        smem[threadIdx.x + 128] = unstable_l3;
        smem[threadIdx.x + 160] = unstable_d4;
        smem[threadIdx.x + 192] = unstable_d5;
        smem[threadIdx.x + 224] = unstable_d6;
    }

    // advance by one generation in-place:
    not_low = gnot_low;
    not_high = gnot_high;
    not_stable = gnot_stable;

    // have we deduced new information about the stable state?
    improvements = hh::ballot_32(improvements != 0);
    return (improvements != 0);
}

