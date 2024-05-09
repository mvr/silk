template<bool Vertical> _DI_ bool apply_domino_rule(uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6) {

    constexpr bool Horizontal = !Vertical;

    // the inputs to this function specify which possibilities have
    // been ruled out (e.g. ad0 == 'we have ruled out dead with 0
    // live neighbours') so forced_live is the conjunction of all
    // of the ad? variables and forced_dead is the conjunction of
    // all of the al? variables:
    uint32_t forced_live = ad0 & ad1 & ad2 & ad4 & ad5 & ad6;
    uint32_t forced_dead = al2 & al3;

    // now we determine whether the cells immediately to the left
    // and right of the current cell are forced live/dead:
    uint32_t flla = kc::shift_plane<Horizontal,  1>(forced_live);
    uint32_t flra = kc::shift_plane<Horizontal, -1>(forced_live);
    uint32_t fdla = kc::shift_plane<Horizontal,  1>(forced_dead);
    uint32_t fdra = kc::shift_plane<Horizontal, -1>(forced_dead);

    // these variables are set to true if we have ruled out a
    // particular number of live cells amongst the cells marked
    // 'x' below:
    //
    // nnn
    // xax
    // xbx
    // nnn
    //
    // where we're in the frame of reference of the cell 'a':
    uint32_t x0, x1, x2, x3, x4;

    {
        // determine which of the x-cells in the row of 'b'
        // are forced to be live/dead:
        uint32_t fllb = kc::shift_plane<Vertical, -1>(flla);
        uint32_t flrb = kc::shift_plane<Vertical, -1>(flra);
        uint32_t fdlb = kc::shift_plane<Vertical, -1>(fdla);
        uint32_t fdrb = kc::shift_plane<Vertical, -1>(fdra);

        // apply a single full-adder to collapse the 4 variables
        // {flla, flra, fllb, flrb} into 3 variables {fl2, fl1, flrb}
        // with respective weights {2, 1, 1}, and do the same for
        // the forced-dead cells:
        uint32_t fl2 = apply_maj3(flla, flra, fllb);
        uint32_t fl1 = apply_xor3(flla, flra, fllb);
        uint32_t fd2 = apply_maj3(fdla, fdra, fdlb);
        uint32_t fd1 = apply_xor3(fdla, fdra, fdlb);

        // determine which counts of x-cells are excluded:
        x0 = (fl2 | (fl1 | flrb));
        x1 = (fl2 | (fl1 & flrb)) | (fd2 & (fd1 & fdrb));
        x2 = (fl2 & (fl1 | flrb)) | (fd2 & (fd1 | fdrb));
        x3 = (fl2 & (fl1 & flrb)) | (fd2 | (fd1 & fdrb));
        x4 =                        (fd2 | (fd1 | fdrb));
    }

    // these variables are set to true if we have ruled out a
    // particular number of live cells amongst the three 'n'
    // cells adjacent to 'a':
    uint32_t an0, an1, an2, an3;

    // and analogously for the 'n' cells adjacent to 'b':
    uint32_t bn0, bn1, bn2, bn3;

    // perform full-adders on the row 'xax':
    uint32_t fl2 = apply_maj3(flla, flra, forced_live);
    uint32_t fl1 = apply_xor3(flla, flra, forced_live);
    uint32_t fd2 = apply_maj3(fdla, fdra, forced_dead);
    uint32_t fd1 = apply_xor3(fdla, fdra, forced_dead);

    {
        // determine which counts of cells are excluded
        // in the row 'xax':
        uint32_t n0 = fl1;
        uint32_t n1 = fl2 | (fd2 & fd1);
        uint32_t n2 = fd2 | (fl2 & fl1);
        uint32_t n3 = fd1;

        // communicate vertically to get the excluded
        // counts in the two rows of 'n' cells:
        an0 = kc::shift_plane<Vertical,  1>(n0);
        an1 = kc::shift_plane<Vertical,  1>(n1);
        an2 = kc::shift_plane<Vertical,  1>(n2);
        an3 = kc::shift_plane<Vertical,  1>(n3);
        bn0 = kc::shift_plane<Vertical, -2>(n0);
        bn1 = kc::shift_plane<Vertical, -2>(n1);
        bn2 = kc::shift_plane<Vertical, -2>(n2);
        bn3 = kc::shift_plane<Vertical, -2>(n3);
    }

    // we deduce information about the neighbour counts of
    // a and b by initialising these variables to 'impossible'
    // and then doing a case-bash over possible configurations
    // of the six cells 'xaxxbx'. Although there are 2**6 = 64
    // combinations, we are invariant under permuting the 'x's
    // which reduces to 5 * 2 * 2 = 20 cases. Moreover, 4 of
    // these are incompatible with CGoL still-lifes so 16 cases
    // remain:
    uint32_t gan0 = 0xffffffffu;
    uint32_t gan1 = 0xffffffffu;
    uint32_t gan2 = 0xffffffffu;
    uint32_t gan3 = 0xffffffffu;
    uint32_t gbn0 = 0xffffffffu;
    uint32_t gbn1 = 0xffffffffu;
    uint32_t gbn2 = 0xffffffffu;
    uint32_t gbn3 = 0xffffffffu;

    uint32_t bd0 = kc::shift_plane<Vertical, -1>(ad0);
    uint32_t gad0 = 0xffffffffu;
    uint32_t gbd0 = 0xffffffffu;

    uint32_t bd1 = kc::shift_plane<Vertical, -1>(ad1);
    uint32_t gad1 = 0xffffffffu;
    uint32_t gbd1 = 0xffffffffu;

    uint32_t bd2 = kc::shift_plane<Vertical, -1>(ad2);
    uint32_t gad2 = 0xffffffffu;
    uint32_t gbd2 = 0xffffffffu;

    uint32_t bl2 = kc::shift_plane<Vertical, -1>(al2);
    uint32_t gal2 = 0xffffffffu;
    uint32_t gbl2 = 0xffffffffu;

    uint32_t bl3 = kc::shift_plane<Vertical, -1>(al3);
    uint32_t gal3 = 0xffffffffu;
    uint32_t gbl3 = 0xffffffffu;

    uint32_t bd4 = kc::shift_plane<Vertical, -1>(ad4);
    uint32_t gad4 = 0xffffffffu;
    uint32_t gbd4 = 0xffffffffu;

    uint32_t bd5 = kc::shift_plane<Vertical, -1>(ad5);
    uint32_t gad5 = 0xffffffffu;
    uint32_t gbd5 = 0xffffffffu;

    uint32_t bd6 = kc::shift_plane<Vertical, -1>(ad6);
    uint32_t gad6 = 0xffffffffu;
    uint32_t gbd6 = 0xffffffffu;

    // now do the 16 cases in turn:
    {
        // x = 0, a = 0, b = 0
        uint32_t contradiction = x0 | ((an0 | ad0) & (an1 | ad1) & (an2 | ad2)) | ((bn0 | bd0) & (bn1 | bd1) & (bn2 | bd2));
        gan0 &= ad0 | contradiction;
        gad0 &= an0 | contradiction;
        gan1 &= ad1 | contradiction;
        gad1 &= an1 | contradiction;
        gan2 &= ad2 | contradiction;
        gad2 &= an2 | contradiction;
        gbn0 &= bd0 | contradiction;
        gbd0 &= bn0 | contradiction;
        gbn1 &= bd1 | contradiction;
        gbd1 &= bn1 | contradiction;
        gbn2 &= bd2 | contradiction;
        gbd2 &= bn2 | contradiction;
    }
    {
        // x = 0, a = 0, b = 1
        uint32_t contradiction = x0 | ((an0 | ad1) & (an1 | ad2) & (an3 | ad4)) | ((bn2 | bl2) & (bn3 | bl3));
        gan0 &= ad1 | contradiction;
        gad1 &= an0 | contradiction;
        gan1 &= ad2 | contradiction;
        gad2 &= an1 | contradiction;
        gan3 &= ad4 | contradiction;
        gad4 &= an3 | contradiction;
        gbn2 &= bl2 | contradiction;
        gbl2 &= bn2 | contradiction;
        gbn3 &= bl3 | contradiction;
        gbl3 &= bn3 | contradiction;
    }
    {
        // x = 0, a = 1, b = 0
        uint32_t contradiction = x0 | ((an2 | al2) & (an3 | al3)) | ((bn0 | bd1) & (bn1 | bd2) & (bn3 | bd4));
        gan2 &= al2 | contradiction;
        gal2 &= an2 | contradiction;
        gan3 &= al3 | contradiction;
        gal3 &= an3 | contradiction;
        gbn0 &= bd1 | contradiction;
        gbd1 &= bn0 | contradiction;
        gbn1 &= bd2 | contradiction;
        gbd2 &= bn1 | contradiction;
        gbn3 &= bd4 | contradiction;
        gbd4 &= bn3 | contradiction;
    }
    {
        // x = 0, a = 1, b = 1
        uint32_t contradiction = x0 | ((an1 | al2) & (an2 | al3)) | ((bn1 | bl2) & (bn2 | bl3));
        gan1 &= al2 | contradiction;
        gal2 &= an1 | contradiction;
        gan2 &= al3 | contradiction;
        gal3 &= an2 | contradiction;
        gbn1 &= bl2 | contradiction;
        gbl2 &= bn1 | contradiction;
        gbn2 &= bl3 | contradiction;
        gbl3 &= bn2 | contradiction;
    }
    {
        // x = 1, a = 0, b = 0
        uint32_t contradiction = x1 | ((an0 | ad1) & (an1 | ad2) & (an3 | ad4)) | ((bn0 | bd1) & (bn1 | bd2) & (bn3 | bd4));
        gan0 &= ad1 | contradiction;
        gad1 &= an0 | contradiction;
        gan1 &= ad2 | contradiction;
        gad2 &= an1 | contradiction;
        gan3 &= ad4 | contradiction;
        gad4 &= an3 | contradiction;
        gbn0 &= bd1 | contradiction;
        gbd1 &= bn0 | contradiction;
        gbn1 &= bd2 | contradiction;
        gbd2 &= bn1 | contradiction;
        gbn3 &= bd4 | contradiction;
        gbd4 &= bn3 | contradiction;
    }
    {
        // x = 1, a = 0, b = 1
        uint32_t contradiction = x1 | ((an0 | ad2) & (an2 | ad4) & (an3 | ad5)) | ((bn1 | bl2) & (bn2 | bl3));
        gan0 &= ad2 | contradiction;
        gad2 &= an0 | contradiction;
        gan2 &= ad4 | contradiction;
        gad4 &= an2 | contradiction;
        gan3 &= ad5 | contradiction;
        gad5 &= an3 | contradiction;
        gbn1 &= bl2 | contradiction;
        gbl2 &= bn1 | contradiction;
        gbn2 &= bl3 | contradiction;
        gbl3 &= bn2 | contradiction;
    }
    {
        // x = 1, a = 1, b = 0
        uint32_t contradiction = x1 | ((an1 | al2) & (an2 | al3)) | ((bn0 | bd2) & (bn2 | bd4) & (bn3 | bd5));
        gan1 &= al2 | contradiction;
        gal2 &= an1 | contradiction;
        gan2 &= al3 | contradiction;
        gal3 &= an2 | contradiction;
        gbn0 &= bd2 | contradiction;
        gbd2 &= bn0 | contradiction;
        gbn2 &= bd4 | contradiction;
        gbd4 &= bn2 | contradiction;
        gbn3 &= bd5 | contradiction;
        gbd5 &= bn3 | contradiction;
    }
    {
        // x = 1, a = 1, b = 1
        uint32_t contradiction = x1 | ((an0 | al2) & (an1 | al3)) | ((bn0 | bl2) & (bn1 | bl3));
        gan0 &= al2 | contradiction;
        gal2 &= an0 | contradiction;
        gan1 &= al3 | contradiction;
        gal3 &= an1 | contradiction;
        gbn0 &= bl2 | contradiction;
        gbl2 &= bn0 | contradiction;
        gbn1 &= bl3 | contradiction;
        gbl3 &= bn1 | contradiction;
    }
    {
        // x = 2, a = 0, b = 0
        uint32_t contradiction = x2 | ((an0 | ad2) & (an2 | ad4) & (an3 | ad5)) | ((bn0 | bd2) & (bn2 | bd4) & (bn3 | bd5));
        gan0 &= ad2 | contradiction;
        gad2 &= an0 | contradiction;
        gan2 &= ad4 | contradiction;
        gad4 &= an2 | contradiction;
        gan3 &= ad5 | contradiction;
        gad5 &= an3 | contradiction;
        gbn0 &= bd2 | contradiction;
        gbd2 &= bn0 | contradiction;
        gbn2 &= bd4 | contradiction;
        gbd4 &= bn2 | contradiction;
        gbn3 &= bd5 | contradiction;
        gbd5 &= bn3 | contradiction;
    }
    {
        // x = 2, a = 0, b = 1
        uint32_t contradiction = x2 | ((an1 | ad4) & (an2 | ad5) & (an3 | ad6)) | ((bn0 | bl2) & (bn1 | bl3));
        gan1 &= ad4 | contradiction;
        gad4 &= an1 | contradiction;
        gan2 &= ad5 | contradiction;
        gad5 &= an2 | contradiction;
        gan3 &= ad6 | contradiction;
        gad6 &= an3 | contradiction;
        gbn0 &= bl2 | contradiction;
        gbl2 &= bn0 | contradiction;
        gbn1 &= bl3 | contradiction;
        gbl3 &= bn1 | contradiction;
    }
    {
        // x = 2, a = 1, b = 0
        uint32_t contradiction = x2 | ((an0 | al2) & (an1 | al3)) | ((bn1 | bd4) & (bn2 | bd5) & (bn3 | bd6));
        gan0 &= al2 | contradiction;
        gal2 &= an0 | contradiction;
        gan1 &= al3 | contradiction;
        gal3 &= an1 | contradiction;
        gbn1 &= bd4 | contradiction;
        gbd4 &= bn1 | contradiction;
        gbn2 &= bd5 | contradiction;
        gbd5 &= bn2 | contradiction;
        gbn3 &= bd6 | contradiction;
        gbd6 &= bn3 | contradiction;
    }
    {
        // x = 2, a = 1, b = 1
        uint32_t contradiction = x2 | ((an0 | al3)) | ((bn0 | bl3));
        gan0 &= al3 | contradiction;
        gal3 &= an0 | contradiction;
        gbn0 &= bl3 | contradiction;
        gbl3 &= bn0 | contradiction;
    }
    {
        // x = 3, a = 0, b = 0
        uint32_t contradiction = x3 | ((an1 | ad4) & (an2 | ad5) & (an3 | ad6)) | ((bn1 | bd4) & (bn2 | bd5) & (bn3 | bd6));
        gan1 &= ad4 | contradiction;
        gad4 &= an1 | contradiction;
        gan2 &= ad5 | contradiction;
        gad5 &= an2 | contradiction;
        gan3 &= ad6 | contradiction;
        gad6 &= an3 | contradiction;
        gbn1 &= bd4 | contradiction;
        gbd4 &= bn1 | contradiction;
        gbn2 &= bd5 | contradiction;
        gbd5 &= bn2 | contradiction;
        gbn3 &= bd6 | contradiction;
        gbd6 &= bn3 | contradiction;
    }
    {
        // x = 3, a = 0, b = 1
        uint32_t contradiction = x3 | ((an0 | ad4) & (an1 | ad5) & (an2 | ad6)) | ((bn0 | bl3));
        gan0 &= ad4 | contradiction;
        gad4 &= an0 | contradiction;
        gan1 &= ad5 | contradiction;
        gad5 &= an1 | contradiction;
        gan2 &= ad6 | contradiction;
        gad6 &= an2 | contradiction;
        gbn0 &= bl3 | contradiction;
        gbl3 &= bn0 | contradiction;
    }
    {
        // x = 3, a = 1, b = 0
        uint32_t contradiction = x3 | ((an0 | al3)) | ((bn0 | bd4) & (bn1 | bd5) & (bn2 | bd6));
        gan0 &= al3 | contradiction;
        gal3 &= an0 | contradiction;
        gbn0 &= bd4 | contradiction;
        gbd4 &= bn0 | contradiction;
        gbn1 &= bd5 | contradiction;
        gbd5 &= bn1 | contradiction;
        gbn2 &= bd6 | contradiction;
        gbd6 &= bn2 | contradiction;
    }
    {
        // x = 4, a = 0, b = 0
        uint32_t contradiction = x4 | ((an0 | ad4) & (an1 | ad5) & (an2 | ad6)) | ((bn0 | bd4) & (bn1 | bd5) & (bn2 | bd6));
        gan0 &= ad4 | contradiction;
        gad4 &= an0 | contradiction;
        gan1 &= ad5 | contradiction;
        gad5 &= an1 | contradiction;
        gan2 &= ad6 | contradiction;
        gad6 &= an2 | contradiction;
        gbn0 &= bd4 | contradiction;
        gbd4 &= bn0 | contradiction;
        gbn1 &= bd5 | contradiction;
        gbd5 &= bn1 | contradiction;
        gbn2 &= bd6 | contradiction;
        gbd6 &= bn2 | contradiction;
    }


    {
        // determine which combinations have been disproved:
        uint32_t gn0 = kc::shift_plane<Vertical, -1>(gan0) | kc::shift_plane<Vertical, 2>(gbn0);
        uint32_t gn1 = kc::shift_plane<Vertical, -1>(gan1) | kc::shift_plane<Vertical, 2>(gbn1);
        uint32_t gn2 = kc::shift_plane<Vertical, -1>(gan2) | kc::shift_plane<Vertical, 2>(gbn2);
        uint32_t gn3 = kc::shift_plane<Vertical, -1>(gan3) | kc::shift_plane<Vertical, 2>(gbn3);

        // determine which combinations have been proved:
        uint32_t gf0 = gn1 & gn2 & gn3;
        uint32_t gf1 = gn0 & gn2 & gn3;
        uint32_t gf2 = gn0 & gn1 & gn3;
        uint32_t gf3 = gn0 & gn1 & gn2;

        // determine whether to propagate live/dead to unknown neighbours:
        uint32_t pl = gf3 | (gf2 & fd1) | (gf1 & fd2);
        uint32_t pd = gf0 | (gf1 & fl2) | (gf2 & fl1);
        uint32_t turn_live = pl | kc::shift_plane<Horizontal, -1>(pl) | kc::shift_plane<Horizontal, 1>(pl);
        uint32_t turn_dead = pd | kc::shift_plane<Horizontal, -1>(pd) | kc::shift_plane<Horizontal, 1>(pd);

        // if unknown, set accordingly:
        forced_live |= (turn_live &~ forced_dead);
        forced_dead |= (turn_dead &~ forced_live);
    }

    // aggregate all of the information that we have learned into
    // the input variables in-place (they are references) whilst
    // tracking whether or not we have made any new deductions:
    uint32_t improvements = 0;
    gad0 |= forced_live | ad0 | kc::shift_plane<Vertical, 1>(gbd0); improvements |= (gad0 &~ ad0); ad0 = gad0;
    gad1 |= forced_live | ad1 | kc::shift_plane<Vertical, 1>(gbd1); improvements |= (gad1 &~ ad1); ad1 = gad1;
    gad2 |= forced_live | ad2 | kc::shift_plane<Vertical, 1>(gbd2); improvements |= (gad2 &~ ad2); ad2 = gad2;
    gal2 |= forced_dead | al2 | kc::shift_plane<Vertical, 1>(gbl2); improvements |= (gal2 &~ al2); al2 = gal2;
    gal3 |= forced_dead | al3 | kc::shift_plane<Vertical, 1>(gbl3); improvements |= (gal3 &~ al3); al3 = gal3;
    gad4 |= forced_live | ad4 | kc::shift_plane<Vertical, 1>(gbd4); improvements |= (gad4 &~ ad4); ad4 = gad4;
    gad5 |= forced_live | ad5 | kc::shift_plane<Vertical, 1>(gbd5); improvements |= (gad5 &~ ad5); ad5 = gad5;
    gad6 |= forced_live | ad6 | kc::shift_plane<Vertical, 1>(gbd6); improvements |= (gad6 &~ ad6); ad6 = gad6;

    improvements = hh::ballot_32(improvements != 0);
    return (improvements != 0);
}

