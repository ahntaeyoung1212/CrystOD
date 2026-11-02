#!/usr/bin/env python3
"""
XDATCAR から ADP (Anisotropic Displacement Parameters) を計算して CIF ファイルを出力する

Usage:
    python xdatcar_to_adp.py [start_step] [--supercell N1 N2 N3] [--xdatcar FILE] [--output FILE]

Example:
    python xdatcar_to_adp.py 20000
    python xdatcar_to_adp.py 20000 --supercell 2 2 2
    python xdatcar_to_adp.py 20000 --supercell 3 3 3 --xdatcar XDATCAR --output ave_with_adp.cif
"""

import sys
import os
import argparse
import numpy as np
from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core import Structure, Lattice
from collections import Counter, defaultdict
import spglib


# =============================================================================
# ADP制約計算用の関数
# =============================================================================

def build_symmetry_projector(rotations):
    """
    サイト対称性の回転行列から6x6射影演算子を構築する
    
    U成分ベクトル [U11, U22, U33, U12, U13, U23] に対する射影演算子を計算
    この射影を適用すると、すべての対称操作で不変なU行列になる
    
    Parameters:
    -----------
    rotations : list of numpy.ndarray
        サイト対称性の回転行列のリスト（分数座標系）
    
    Returns:
    --------
    numpy.ndarray (6, 6)
        射影演算子行列
    """
    # U成分のインデックス: [U11, U22, U33, U12, U13, U23]
    u_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    
    # 平均変換行列を計算
    avg_transform = np.zeros((6, 6))
    
    for R_frac in rotations:
        # 各U成分基底の変換を計算
        transform = np.zeros((6, 6))
        
        for src_idx, (i, j) in enumerate(u_indices):
            # 元のU成分だけが1の対称行列を作成
            U_basis = np.zeros((3, 3))
            U_basis[i, j] = 1.0
            U_basis[j, i] = 1.0
            
            # 変換: U' = R @ U @ R.T
            U_transformed = R_frac @ U_basis @ R_frac.T
            
            # 変換後の各成分への寄与を計算
            for dst_idx, (di, dj) in enumerate(u_indices):
                transform[dst_idx, src_idx] = U_transformed[di, dj]
        
        avg_transform += transform
    
    # 平均を取る（射影演算子）
    projector = avg_transform / len(rotations)
    
    return projector


def apply_symmetry_constraints(U_cryst, projector):
    """
    射影演算子を使用してU行列に対称性制約を適用する
    
    Parameters:
    -----------
    U_cryst : numpy.ndarray (3, 3)
        結晶座標系でのADP行列
    projector : numpy.ndarray (6, 6)
        射影演算子
    
    Returns:
    --------
    numpy.ndarray (3, 3)
        対称性制約を満たすADP行列
    """
    # U行列をベクトル化 [U11, U22, U33, U12, U13, U23]
    u_vec = np.array([
        U_cryst[0, 0],  # U11
        U_cryst[1, 1],  # U22
        U_cryst[2, 2],  # U33
        U_cryst[0, 1],  # U12
        U_cryst[0, 2],  # U13
        U_cryst[1, 2],  # U23
    ])
    
    # 射影を適用
    u_projected = projector @ u_vec
    
    # 行列に戻す
    U_constrained = np.array([
        [u_projected[0], u_projected[3], u_projected[4]],
        [u_projected[3], u_projected[1], u_projected[5]],
        [u_projected[4], u_projected[5], u_projected[2]]
    ])
    
    return U_constrained


def get_constraint_description(projector, tol=1e-6):
    """
    射影演算子から制約の説明文字列を生成する
    
    Returns:
    --------
    str
        制約の説明（例: "U11=U22, U13=-U23"）
    """
    u_names = ['U11', 'U22', 'U33', 'U12', 'U13', 'U23']
    constraints = []
    
    # 各成分について、射影後の値を確認
    for i in range(6):
        # 単位ベクトル
        e_i = np.zeros(6)
        e_i[i] = 1.0
        
        # 射影
        p_i = projector @ e_i
        
        # ゼロ制約をチェック
        if abs(p_i[i]) < tol:
            constraints.append(f"{u_names[i]}=0")
        # 等式制約をチェック
        else:
            for j in range(i + 1, 6):
                if abs(p_i[j]) > tol:
                    # 符号を確認
                    if abs(p_i[j] - p_i[i]) < tol:
                        constraints.append(f"{u_names[i]}={u_names[j]}")
                    elif abs(p_i[j] + p_i[i]) < tol:
                        constraints.append(f"{u_names[i]}=-{u_names[j]}")
    
    return ", ".join(constraints) if constraints else "制約なし"


def get_site_symmetry_operations(coords, rotations, translations, symprec=0.1):
    """
    指定した座標を不変にする対称操作を取得する
    """
    site_sym_ops = []
    
    for R, t in zip(rotations, translations):
        transformed = (R @ coords + t) % 1.0
        diff = transformed - coords
        diff = diff - np.round(diff)
        
        if np.linalg.norm(diff) < symprec:
            site_sym_ops.append(R)
    
    return site_sym_ops


def rotation_to_xyz(R, t):
    """回転行列と並進ベクトルをCIF形式のxyz文字列に変換する"""
    axes = ['x', 'y', 'z']
    result = []
    for i in range(3):
        terms = []
        for j in range(3):
            if R[i, j] == 1:
                terms.append(axes[j])
            elif R[i, j] == -1:
                terms.append(f"-{axes[j]}")
            elif R[i, j] != 0:
                terms.append(f"{R[i, j]}*{axes[j]}")
        
        if abs(t[i]) > 1e-6:
            frac = t[i]
            if abs(frac - 0.5) < 1e-6:
                terms.append("1/2")
            elif abs(frac - 0.25) < 1e-6:
                terms.append("1/4")
            elif abs(frac - 0.75) < 1e-6:
                terms.append("3/4")
            elif abs(frac - 1/3) < 1e-6:
                terms.append("1/3")
            elif abs(frac - 2/3) < 1e-6:
                terms.append("2/3")
            else:
                terms.append(f"{frac:.4f}")
        
        if not terms:
            terms = ['0']
        
        result.append('+'.join(terms).replace('+-', '-'))
    
    return ', '.join(result)


# =============================================================================
# メイン処理
# =============================================================================

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='XDATCAR から ADP を計算して CIF ファイルを出力する',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python xdatcar_to_adp.py 20000
    python xdatcar_to_adp.py 20000 --supercell 2 2 2
    python xdatcar_to_adp.py 20000 --supercell 3 3 3 --xdatcar XDATCAR --output ave_with_adp.cif
        '''
    )
    parser.add_argument('start_step', type=int, nargs='?', default=0,
                        help='解析開始ステップ (default: 0)')
    parser.add_argument('--supercell', '-s', type=int, nargs=3, default=[2, 2, 2],
                        metavar=('N1', 'N2', 'N3'),
                        help='スーパーセルサイズ (対角成分のみ、default: 2 2 2)')
    parser.add_argument('--xdatcar', '-x', type=str, default='XDATCAR',
                        help='入力XDATCARファイル名 (default: XDATCAR)')
    parser.add_argument('--output', '-o', type=str, default='ave_with_adp.cif',
                        help='出力CIFファイル名 (default: ave_with_adp.cif)')
    parser.add_argument('--symprec', type=float, default=0.1,
                        help='対称性判定の許容誤差 (default: 0.1)')
    parser.add_argument('--grouping-tolerance', type=float, default=0.1,
                        help='グルーピングの許容誤差 (default: 0.1)')
    
    args = parser.parse_args()
    
    # 設定
    SUPERCELL_SIZE = np.array(args.supercell)
    XDATCAR_FILE = args.xdatcar
    OUTPUT_CIF_FILE = args.output
    start_step = args.start_step
    SYMPREC = args.symprec
    GROUPING_TOLERANCE = args.grouping_tolerance
    
    print(f"スーパーセルサイズ: {SUPERCELL_SIZE}")
    print(f"開始ステップ: {start_step}")
    print(f"入力ファイル: {XDATCAR_FILE}")
    print(f"出力ファイル: {OUTPUT_CIF_FILE}")
    
    # =========================================================================
    # ステップ1: XDATCARの読み込み
    # =========================================================================
    print("\nXDATCARを読み込み中...（時間がかかります）")
    xdatcar = Xdatcar(XDATCAR_FILE)
    all_structures = xdatcar.structures
    
    structures_to_process = all_structures[start_step:]
    n_steps = len(structures_to_process)
    
    first_structure = structures_to_process[0]
    chem_formula = [site.specie.symbol for site in first_structure.sites]
    lattice_vec_super = np.array(first_structure.lattice.matrix)
    n_atoms_super = len(chem_formula)
    
    print(f"\nスーパーセル情報:")
    print(f"  原子数: {n_atoms_super}")
    print(f"  組成: {first_structure.composition}")
    print(f"  解析ステップ数: {n_steps}")
    
    lattice_vec_unit = lattice_vec_super / SUPERCELL_SIZE[:, np.newaxis]
    
    coordinates = []
    for structure in structures_to_process:
        coords_list = [site.frac_coords for site in structure.sites]
        coordinates.append(coords_list)
    coordinates = np.array(coordinates)
    
    print(f"  座標配列の形状: {coordinates.shape}")
    
    # =========================================================================
    # ステップ2: 時間平均座標
    # =========================================================================
    avg_coords = np.zeros((n_atoms_super, 3))
    
    for i in range(n_atoms_super):
        atom_coords = coordinates[:, i, :]
        for axis in range(3):
            angles = atom_coords[:, axis] * 2 * np.pi
            mean_cos = np.mean(np.cos(angles))
            mean_sin = np.mean(np.sin(angles))
            avg_angle = np.arctan2(mean_sin, mean_cos)
            avg_coords[i, axis] = (avg_angle / (2 * np.pi)) % 1.0
    
    # =========================================================================
    # ステップ3: 座標の畳み込み
    # =========================================================================
    fold_thresholds = 1.0 / SUPERCELL_SIZE
    folded_coords = avg_coords.copy()
    for axis in range(3):
        threshold = fold_thresholds[axis]
        folded_coords[:, axis] = folded_coords[:, axis] % threshold
    
    # =========================================================================
    # ステップ4: グルーピング
    # =========================================================================
    grouped_indices = []
    remaining_indices = set(range(n_atoms_super))
    adjusted_tolerance = GROUPING_TOLERANCE / np.mean(SUPERCELL_SIZE)
    
    while remaining_indices:
        ref_idx = min(remaining_indices)
        ref_coord = folded_coords[ref_idx]
        ref_element = chem_formula[ref_idx]
        
        group = []
        for idx in list(remaining_indices):
            if chem_formula[idx] != ref_element:
                continue
            
            test_coord = folded_coords[idx]
            diff = ref_coord - test_coord
            
            for axis in range(3):
                threshold = fold_thresholds[axis]
                diff[axis] = diff[axis] - np.round(diff[axis] / threshold) * threshold
            
            dist = np.linalg.norm(diff)
            
            if dist < adjusted_tolerance:
                group.append(idx)
                remaining_indices.remove(idx)
        
        grouped_indices.append(group)
    
    n_groups = len(grouped_indices)
    print(f"\nグループ数: {n_groups}")
    print(f"期待されるレプリカ数/グループ: {int(np.prod(SUPERCELL_SIZE))}")
    
    # =========================================================================
    # ステップ5: グループの平均座標
    # =========================================================================
    group_unitcell_coords = []
    group_elements = []
    
    for group_idx, atom_indices_in_group in enumerate(grouped_indices):
        element = chem_formula[atom_indices_in_group[0]]
        group_elements.append(element)
        
        group_coords_folded = folded_coords[atom_indices_in_group]
        
        avg_coord_folded = np.zeros(3)
        for axis in range(3):
            threshold = fold_thresholds[axis]
            angles = group_coords_folded[:, axis] * 2 * np.pi / threshold
            mean_cos = np.mean(np.cos(angles))
            mean_sin = np.mean(np.sin(angles))
            avg_angle = np.arctan2(mean_sin, mean_cos)
            avg_coord_folded[axis] = ((avg_angle / (2 * np.pi)) * threshold) % threshold
        
        unitcell_coord = (avg_coord_folded * SUPERCELL_SIZE) % 1.0
        group_unitcell_coords.append(unitcell_coord)
    
    group_unitcell_coords = np.array(group_unitcell_coords)
    
    # =========================================================================
    # ステップ7: ユニットセル構造の作成と対称性解析
    # =========================================================================
    # ユニットセル構造を作成
    unitcell_structure = Structure(
        lattice_vec_unit,
        group_elements,
        group_unitcell_coords,
        coords_are_cartesian=False
    )
    
    # 元素番号のマッピング
    unique_elements = sorted(set(group_elements))
    element_to_number = {el: i+1 for i, el in enumerate(unique_elements)}
    atom_numbers = [element_to_number[el] for el in group_elements]
    
    # spglibで対称性を解析
    cell = (lattice_vec_unit, group_unitcell_coords, atom_numbers)
    symmetry_dataset = spglib.get_symmetry_dataset(cell, symprec=SYMPREC)
    
    if symmetry_dataset is None:
        raise ValueError("spglibで対称性が見つかりませんでした")
    
    # 属性アクセスを使用
    spacegroup = symmetry_dataset.international
    spacegroup_number = symmetry_dataset.number
    rotations = symmetry_dataset.rotations
    translations = symmetry_dataset.translations
    equivalent_atoms = symmetry_dataset.equivalent_atoms
    
    print(f"\n空間群: {spacegroup} (No. {spacegroup_number})")
    print(f"ユニットセル内原子数: {len(unitcell_structure)}")
    print(f"対称操作の数: {len(rotations)}")
    
    # 非対称単位のサイトを取得
    wyckoff_ids_sorted = sorted(set(equivalent_atoms))
    n_asym_sites = len(wyckoff_ids_sorted)
    
    print(f"非対称単位のサイト数: {n_asym_sites}")
    
    # Wyckoff IDからCIF出力用のインデックスへのマッピング
    wyckoff_to_cif_idx = {wid: i for i, wid in enumerate(wyckoff_ids_sorted)}
    
    # 非対称単位の代表原子情報を取得
    asym_sites = []
    for cif_idx, wyckoff_id in enumerate(wyckoff_ids_sorted):
        element = group_elements[wyckoff_id]
        coords = group_unitcell_coords[wyckoff_id]
        multiplicity = np.sum(equivalent_atoms == wyckoff_id)
        asym_sites.append({
            'label': f"{element}{cif_idx}",
            'element': element,
            'frac_coords': coords,
            'multiplicity': multiplicity,
            'wyckoff_id': wyckoff_id
        })
        print(f"  {asym_sites[-1]['label']}: mult={multiplicity}, coords=({coords[0]:.5f}, {coords[1]:.5f}, {coords[2]:.5f})")
    
    # =========================================================================
    # ADP制約の計算
    # =========================================================================
    wyckoff_constraints = {}
    wyckoff_projectors = {}
    
    print("\nWyckoff位置ごとのADP制約:")
    for wyckoff_id in sorted(set(equivalent_atoms)):
        coords = group_unitcell_coords[wyckoff_id]
        site_sym_ops = get_site_symmetry_operations(coords, rotations, translations, SYMPREC)
        
        # 射影演算子を構築
        projector = build_symmetry_projector(site_sym_ops)
        wyckoff_projectors[wyckoff_id] = projector
        
        # 制約の説明を取得
        constraint_str = get_constraint_description(projector)
        wyckoff_constraints[wyckoff_id] = constraint_str
        
        element = group_elements[wyckoff_id]
        print(f"  Wyckoff位置 {wyckoff_id} ({element}): サイト対称性次数={len(site_sym_ops)}, {constraint_str}")
    
    # =========================================================================
    # ステップ9: 座標変換
    # =========================================================================
    all_atom_cart_coords = []
    all_atom_group_ids = []
    
    lattice_inv = np.linalg.inv(lattice_vec_unit)
    
    for group_idx, atom_indices in enumerate(grouped_indices):
        for atom_idx in atom_indices:
            unitcell_frac_unwrapped = None
            atom_coords_all_steps = []
            
            for step in range(n_steps):
                supercell_frac = coordinates[step, atom_idx, :].copy()
                supercell_frac = supercell_frac % 1.0
                supercell_cart = supercell_frac @ lattice_vec_super
                unitcell_frac = supercell_cart @ lattice_inv
                
                if step == 0:
                    unitcell_frac_unwrapped = unitcell_frac - np.round(unitcell_frac)
                else:
                    diff = unitcell_frac - unitcell_frac_unwrapped
                    diff = diff - np.round(diff)
                    unitcell_frac_unwrapped = unitcell_frac_unwrapped + diff
                
                unitcell_cart = unitcell_frac_unwrapped @ lattice_vec_unit
                atom_coords_all_steps.append(unitcell_cart)
            
            all_atom_cart_coords.append(np.array(atom_coords_all_steps))
            all_atom_group_ids.append(group_idx)
    
    all_atom_cart_coords = np.array(all_atom_cart_coords)
    all_atom_group_ids = np.array(all_atom_group_ids)
    
    n_atoms_supercell = len(all_atom_cart_coords)
    print(f"\n座標変換完了: {n_atoms_supercell}原子 × {n_steps}ステップ")
    
    # =========================================================================
    # ステップ10: Wyckoff位置と対称操作の割り当て
    # =========================================================================
    all_atom_wyckoff_ids = []
    all_atom_symop_ids = []
    
    for atom_index in range(n_atoms_supercell):
        group_idx = all_atom_group_ids[atom_index]
        wyckoff_id = equivalent_atoms[group_idx]
        all_atom_wyckoff_ids.append(wyckoff_id)
        
        coord = group_unitcell_coords[group_idx]
        representative_coord = group_unitcell_coords[wyckoff_id]
        
        found_symop = None
        for symop_idx in range(len(rotations)):
            R = rotations[symop_idx]
            t = translations[symop_idx]
            transformed_coord = (R @ coord + t) % 1.0
            diff = transformed_coord - representative_coord
            diff = diff - np.round(diff)
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                found_symop = symop_idx
                break
        
        all_atom_symop_ids.append(found_symop)
    
    all_atom_wyckoff_ids = np.array(all_atom_wyckoff_ids)
    all_atom_symop_ids = np.array(all_atom_symop_ids)
    
    print(f"Wyckoff位置割り当て完了")
    
    # =========================================================================
    # ステップ11: 変位計算
    # =========================================================================
    volume = np.linalg.det(lattice_vec_unit)
    a_star = np.cross(lattice_vec_unit[1], lattice_vec_unit[2]) / volume
    b_star = np.cross(lattice_vec_unit[2], lattice_vec_unit[0]) / volume
    c_star = np.cross(lattice_vec_unit[0], lattice_vec_unit[1]) / volume
    
    A = np.array([
        a_star / np.linalg.norm(a_star),
        b_star / np.linalg.norm(b_star),
        c_star / np.linalg.norm(c_star)
    ])
    
    wyckoff_displacements = defaultdict(list)
    
    M = lattice_vec_unit
    M_inv = np.linalg.inv(M)
    
    for atom_idx in range(n_atoms_supercell):
        wyckoff_id = all_atom_wyckoff_ids[atom_idx]
        symop_idx = all_atom_symop_ids[atom_idx]
        
        if symop_idx is None:
            continue
        
        atom_cart_coords = all_atom_cart_coords[atom_idx]
        mean_coord = np.mean(atom_cart_coords, axis=0)
        displacements = atom_cart_coords - mean_coord
        
        R_frac = rotations[symop_idx]
        R_cart = M @ R_frac @ M_inv
        displacements_transformed = displacements @ R_cart.T
        
        wyckoff_displacements[wyckoff_id].append(displacements_transformed)
    
    for wyckoff_id in list(wyckoff_displacements.keys()):
        wyckoff_displacements[wyckoff_id] = np.concatenate(
            wyckoff_displacements[wyckoff_id], axis=0
        )
    
    print(f"変位データ収集完了")
    
    # =========================================================================
    # ステップ12: ADP計算
    # =========================================================================
    wyckoff_ids_sorted = sorted(wyckoff_displacements.keys())
    
    wyckoff_averaged_U_cryst = {}
    wyckoff_averaged_U_cart = {}
    
    print(f"\n{'サイト':<10} {'Ueq (Å²)':<12} {'制約':<35}")
    print("-"*60)
    
    for wyckoff_id in wyckoff_ids_sorted:
        displacements_all = wyckoff_displacements[wyckoff_id]
        
        U_cart = np.cov(displacements_all.T, bias=True)
        U_cryst_before = A @ U_cart @ A.T
        
        # 射影演算子を使用して対称性制約を適用
        projector = wyckoff_projectors[wyckoff_id]
        U_cryst = apply_symmetry_constraints(U_cryst_before, projector)
        
        # 修正前後の値を表示
        cif_idx = wyckoff_to_cif_idx[wyckoff_id]
        site_info = asym_sites[cif_idx]
        label = site_info['label']
        constraint_str = wyckoff_constraints[wyckoff_id]
        
        # 値に変化があった場合のみ詳細を表示
        u_before = np.array([U_cryst_before[0,0], U_cryst_before[1,1], U_cryst_before[2,2],
                            U_cryst_before[0,1], U_cryst_before[0,2], U_cryst_before[1,2]])
        u_after = np.array([U_cryst[0,0], U_cryst[1,1], U_cryst[2,2],
                           U_cryst[0,1], U_cryst[0,2], U_cryst[1,2]])
        max_diff = np.max(np.abs(u_after - u_before))
        
        if max_diff > 1e-6:
            print(f"\n{label} - 対称性制約適用 ({constraint_str}):")
            print(f"  修正前: U11={U_cryst_before[0,0]:.6f} U22={U_cryst_before[1,1]:.6f} U33={U_cryst_before[2,2]:.6f}")
            print(f"         U12={U_cryst_before[0,1]:.6f} U13={U_cryst_before[0,2]:.6f} U23={U_cryst_before[1,2]:.6f}")
            print(f"  修正後: U11={U_cryst[0,0]:.6f} U22={U_cryst[1,1]:.6f} U33={U_cryst[2,2]:.6f}")
            print(f"         U12={U_cryst[0,1]:.6f} U13={U_cryst[0,2]:.6f} U23={U_cryst[1,2]:.6f}")
            print(f"  最大変化量: {max_diff:.6e} Å²")
        
        wyckoff_averaged_U_cryst[wyckoff_id] = U_cryst
        wyckoff_averaged_U_cart[wyckoff_id] = U_cart
        
        eigenvalues, _ = np.linalg.eig(U_cart)
        Ueq = np.mean(np.real(eigenvalues))
        
        print(f"{label:<10} {Ueq:<12.6f} {constraint_str:<35}")
    
    # =========================================================================
    # ステップ13: CIFファイル出力
    # =========================================================================
    # 格子定数
    lattice = Lattice(lattice_vec_unit)
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    
    # 組成式
    element_counts = Counter(group_elements)
    formula = ''.join([f"{el}{count}" if count > 1 else el for el, count in sorted(element_counts.items())])
    formula_sum = ' '.join([f"{el}{count}" for el, count in sorted(element_counts.items())])
    
    # CIFコンテンツ作成
    cif_content = []
    cif_content.append(f"# generated using xdatcar_to_adp.py")
    cif_content.append(f"data_{formula}")
    cif_content.append(f"_symmetry_space_group_name_H-M   '{spacegroup}'")
    cif_content.append(f"_cell_length_a   {a:.8f}")
    cif_content.append(f"_cell_length_b   {b:.8f}")
    cif_content.append(f"_cell_length_c   {c:.8f}")
    cif_content.append(f"_cell_angle_alpha   {alpha:.8f}")
    cif_content.append(f"_cell_angle_beta   {beta:.8f}")
    cif_content.append(f"_cell_angle_gamma   {gamma:.8f}")
    cif_content.append(f"_symmetry_Int_Tables_number   {spacegroup_number}")
    cif_content.append(f"_chemical_formula_structural   {formula}")
    cif_content.append(f"_chemical_formula_sum   '{formula_sum}'")
    cif_content.append(f"_cell_volume   {lattice.volume:.8f}")
    cif_content.append(f"_cell_formula_units_Z   1")
    cif_content.append("")
    
    # 対称操作
    cif_content.append("loop_")
    cif_content.append(" _symmetry_equiv_pos_site_id")
    cif_content.append(" _symmetry_equiv_pos_as_xyz")
    for i, (R, t) in enumerate(zip(rotations, translations)):
        xyz_str = rotation_to_xyz(R, t)
        cif_content.append(f"  {i+1}  '{xyz_str}'")
    cif_content.append("")
    
    # 原子座標（非対称単位のみ）
    cif_content.append("loop_")
    cif_content.append(" _atom_site_type_symbol")
    cif_content.append(" _atom_site_label")
    cif_content.append(" _atom_site_symmetry_multiplicity")
    cif_content.append(" _atom_site_fract_x")
    cif_content.append(" _atom_site_fract_y")
    cif_content.append(" _atom_site_fract_z")
    cif_content.append(" _atom_site_occupancy")
    
    for wyckoff_id in wyckoff_ids_sorted:
        cif_idx = wyckoff_to_cif_idx[wyckoff_id]
        site_info = asym_sites[cif_idx]
        label = site_info['label']
        element = site_info['element']
        coords = site_info['frac_coords']
        mult = site_info['multiplicity']
        
        cif_content.append(f"  {element}  {label}  {mult}  {coords[0]:.8f}  {coords[1]:.8f}  {coords[2]:.8f}  1")
    cif_content.append("")
    
    # ADP情報
    cif_content.append("loop_")
    cif_content.append(" _atom_site_aniso_label")
    cif_content.append(" _atom_site_aniso_U_11")
    cif_content.append(" _atom_site_aniso_U_22")
    cif_content.append(" _atom_site_aniso_U_33")
    cif_content.append(" _atom_site_aniso_U_23")
    cif_content.append(" _atom_site_aniso_U_13")
    cif_content.append(" _atom_site_aniso_U_12")
    
    for wyckoff_id in wyckoff_ids_sorted:
        cif_idx = wyckoff_to_cif_idx[wyckoff_id]
        U_cryst = wyckoff_averaged_U_cryst[wyckoff_id]
        label = asym_sites[cif_idx]['label']
        
        cif_content.append(f" {label:<8}{U_cryst[0,0]:<10.5f}{U_cryst[1,1]:<10.5f}{U_cryst[2,2]:<10.5f}{U_cryst[1,2]:<10.5f}{U_cryst[0,2]:<10.5f}{U_cryst[0,1]:<10.5f}")
    
    # ファイルに書き込み
    with open(OUTPUT_CIF_FILE, 'w') as f:
        f.write('\n'.join(cif_content))
    
    print(f"\n'{OUTPUT_CIF_FILE}' に保存しました")
    print(f"非対称単位のサイト数: {len(wyckoff_ids_sorted)}")


if __name__ == '__main__':
    main()
