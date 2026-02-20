#!/usr/bin/env python3
"""
Laborator 2: Procesarea datelor PCAP - De la pachete la date tabelare

Acest script demonstrează cum să:
1. Citim fișiere PCAP folosind Scapy
2. Extragem features relevante din pachete
3. Agregăm pachetele în flows
4. Exportăm datele într-un format CSV pentru Machine Learning
5. Fiecare rând = pachet(Packets); Fiecare rând = conversație între 2 end-pointuri, conține date pentru algoritmii de ML, deoarece se văd comportamente, nu pachete izolate
6. Pachet = glonț; Flow = schimb de focuri 

Dependențe: pip install scapy pandas
"""

import os
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("EROARE: Scapy nu este instalat!")
    print("Instalați cu: pip install scapy")
    sys.exit(1)


# ============================================================================
# SECȚIUNEA 1: Citirea și explorarea pachetelor
# ============================================================================

def load_pcap(filepath):
    """
    Încarcă un fișier PCAP și returnează lista de pachete.

    Args:
        filepath: Calea către fișierul PCAP

    Returns:
        Lista de pachete Scapy
    """
    print(f"[*] Încărcare fișier: {filepath}")
    packets = rdpcap(filepath)
    print(f"[+] Încărcate {len(packets)} pachete")
    return packets


def explore_packets(packets, num_packets):
    print(f"\n[*] Explorare pachete (max {num_packets} afișate):\n")

    shown = 0
    for i, pkt in enumerate(packets):
        # afișăm doar pachete IP care au TCP sau UDP
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            continue

        print(f"--- Pachet #{i+1} ---")
        print(f"Sumar: {pkt.summary()}")

        print(f"  IP Sursă: {pkt[IP].src}")
        print(f"  IP Destinație: {pkt[IP].dst}")
        print(f"  Protocol IP (număr): {pkt[IP].proto}")
        print(f"  TTL: {pkt[IP].ttl}")
        print(f"  Lungime totală (IP.len): {pkt[IP].len}")

        if TCP in pkt:
            print("  [TCP]")
            print(f"  Port Sursă: {pkt[TCP].sport}")
            print(f"  Port Destinație: {pkt[TCP].dport}")
            print(f"  Flags TCP: {pkt[TCP].flags}")
        elif UDP in pkt:
            print("  [UDP]")
            print(f"  Port Sursă: {pkt[UDP].sport}")
            print(f"  Port Destinație: {pkt[UDP].dport}")
            print(f"  Lungime UDP: {pkt[UDP].len}")

        print()
        shown += 1
        if shown >= num_packets:
            break

    if shown == 0:
        print("Nu am găsit pachete TCP/UDP în captura dată (în filtrele aplicate).")



# ============================================================================
# SECȚIUNEA 2: Extragerea features din pachete individuale
# ============================================================================

def extract_packet_features(packet):
    """
    Extrage features relevante dintr-un pachet individual.

    Args:
        packet: Pachet Scapy

    Returns:
        Dictionary cu features extrase sau None dacă pachetul nu e valid
    """
    features = {}

    # Verificăm dacă pachetul are layer IP
    if IP not in packet:
        return None

    ip_layer = packet[IP]

    # Features de bază IP
    features['ip_src'] = ip_layer.src
    features['ip_dst'] = ip_layer.dst
    features['ip_proto'] = ip_layer.proto
    features['ip_ttl'] = ip_layer.ttl
    features['ip_len'] = ip_layer.len

    # Timestamp (dacă e disponibil)
    features['timestamp'] = float(packet.time) if hasattr(packet, 'time') else 0

    # Features TCP
    if TCP in packet:
        tcp_layer = packet[TCP]
        features['protocol'] = 'TCP'
        features['src_port'] = tcp_layer.sport
        features['dst_port'] = tcp_layer.dport
        features['tcp_flags'] = str(tcp_layer.flags)
        features['tcp_seq'] = tcp_layer.seq
        features['tcp_ack'] = tcp_layer.ack
        features['tcp_window'] = tcp_layer.window

        # Decodificăm flags-urile TCP individual
        flags = tcp_layer.flags
        features['flag_syn'] = 1 if 'S' in str(flags) else 0
        features['flag_ack'] = 1 if 'A' in str(flags) else 0
        features['flag_fin'] = 1 if 'F' in str(flags) else 0
        features['flag_rst'] = 1 if 'R' in str(flags) else 0
        features['flag_psh'] = 1 if 'P' in str(flags) else 0
        features['flag_urg'] = 1 if 'U' in str(flags) else 0

    # Features UDP
    elif UDP in packet:
        udp_layer = packet[UDP]
        features['protocol'] = 'UDP'
        features['src_port'] = udp_layer.sport
        features['dst_port'] = udp_layer.dport
        features['udp_len'] = udp_layer.len

        # UDP nu are flags
        features['tcp_flags'] = ''
        features['tcp_seq'] = 0
        features['tcp_ack'] = 0
        features['tcp_window'] = 0
        features['flag_syn'] = 0
        features['flag_ack'] = 0
        features['flag_fin'] = 0
        features['flag_rst'] = 0
        features['flag_psh'] = 0
        features['flag_urg'] = 0

    # ICMP sau alt protocol
    elif ICMP in packet:
        features['protocol'] = 'ICMP'
        features['src_port'] = 0
        features['dst_port'] = 0
        features['tcp_flags'] = ''
        features['tcp_seq'] = 0
        features['tcp_ack'] = 0
        features['tcp_window'] = 0
        features['flag_syn'] = 0
        features['flag_ack'] = 0
        features['flag_fin'] = 0
        features['flag_rst'] = 0
        features['flag_psh'] = 0
        features['flag_urg'] = 0

    else:
        features['protocol'] = 'OTHER'
        features['src_port'] = 0
        features['dst_port'] = 0
        features['tcp_flags'] = ''
        features['tcp_seq'] = 0
        features['tcp_ack'] = 0
        features['tcp_window'] = 0
        features['flag_syn'] = 0
        features['flag_ack'] = 0
        features['flag_fin'] = 0
        features['flag_rst'] = 0
        features['flag_psh'] = 0
        features['flag_urg'] = 0

    features['payload_size'] = len(packet.payload)

    return features


def packets_to_dataframe(packets):
    """
    Convertește o listă de pachete într-un DataFrame pandas.

    Args:
        packets: Lista de pachete Scapy

    Returns:
        DataFrame pandas cu features extrase
    """
    print(f"\n[*] Extragere features din {len(packets)} pachete...")

    features_list = []
    for i, pkt in enumerate(packets):
        features = extract_packet_features(pkt)
        if features:
            features_list.append(features)

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"    Procesate {i + 1}/{len(packets)} pachete...")

    df = pd.DataFrame(features_list)
    print(f"[+] Creat DataFrame cu {len(df)} rânduri și {len(df.columns)} coloane")

    return df


# ============================================================================
# SECȚIUNEA 3: Agregarea pachetelor în flows
# ============================================================================

def get_flow_key(packet):
    """
    Generează cheia unică pentru un flow (5-tuple).

    Un flow este identificat unic prin:
    - IP sursă
    - IP destinație
    - Port sursă
    - Port destinație
    - Protocol

    Args:
        packet: Pachet Scapy

    Returns:
        Tuple reprezentând flow key sau None
    """
    if IP not in packet:
        return None

    ip_src = packet[IP].src
    ip_dst = packet[IP].dst

    if TCP in packet:
        protocol = 'TCP'
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        protocol = 'UDP'
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    else:
        protocol = 'OTHER'
        src_port = 0
        dst_port = 0

    # Creăm cheia bidirecțională (A->B și B->A sunt același flow)
    if (ip_src, src_port) < (ip_dst, dst_port):
        return (ip_src, ip_dst, src_port, dst_port, protocol)
    else:
        return (ip_dst, ip_src, dst_port, src_port, protocol)


def aggregate_flows(packets):
    """
    Agregă pachetele în flows și calculează statistici pentru fiecare flow.

    Args:
        packets: Lista de pachete Scapy

    Returns:
        DataFrame pandas cu flow features
    """
    print(f"\n[*] Agregare {len(packets)} pachete în flows...")

    # Dictionary pentru a stoca pachetele fiecărui flow
    flows = defaultdict(list)

    for pkt in packets:
        flow_key = get_flow_key(pkt)
        if flow_key:
            flows[flow_key].append(pkt)

    print(f"[+] Identificate {len(flows)} flows unice")

    # Calculăm statistici pentru fiecare flow
    flow_features = []

    for flow_key, flow_packets in flows.items():
        ip_src, ip_dst, src_port, dst_port, protocol = flow_key

        # Timestamps
        timestamps = [float(pkt.time) for pkt in flow_packets if hasattr(pkt, 'time')]
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration = end_time - start_time
        else:
            start_time = end_time = duration = 0

        # Statistici pachete
        packet_count = len(flow_packets)
        total_bytes = sum(pkt[IP].len for pkt in flow_packets if IP in pkt)

        # Statistici forward/backward (simplificat)
        fwd_packets = [pkt for pkt in flow_packets
                       if IP in pkt and pkt[IP].src == ip_src]
        bwd_packets = [pkt for pkt in flow_packets
                       if IP in pkt and pkt[IP].src == ip_dst]

        fwd_count = len(fwd_packets)
        bwd_count = len(bwd_packets)

        fwd_bytes = sum(pkt[IP].len for pkt in fwd_packets if IP in pkt)
        bwd_bytes = sum(pkt[IP].len for pkt in bwd_packets if IP in pkt)

        # Statistici TCP flags (doar pentru TCP)
        syn_count = 0
        ack_count = 0
        fin_count = 0
        rst_count = 0

        if protocol == 'TCP':
            for pkt in flow_packets:
                if TCP in pkt:
                    flags = str(pkt[TCP].flags)
                    if 'S' in flags:
                        syn_count += 1
                    if 'A' in flags:
                        ack_count += 1
                    if 'F' in flags:
                        fin_count += 1
                    if 'R' in flags:
                        rst_count += 1

        # Calculăm rate-uri
        if duration > 0:
            packets_per_second = packet_count / duration
            bytes_per_second = total_bytes / duration
        else:
            packets_per_second = packet_count
            bytes_per_second = total_bytes

        # Media dimensiunii pachetelor
        avg_packet_size = total_bytes / packet_count if packet_count > 0 else 0

        # Creăm feature vector pentru acest flow
        flow_feat = {
            'ip_src': ip_src,
            'ip_dst': ip_dst,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'duration': duration,
            'packet_count': packet_count,
            'total_bytes': total_bytes,
            'fwd_packets': fwd_count,
            'bwd_packets': bwd_count,
            'fwd_bytes': fwd_bytes,
            'bwd_bytes': bwd_bytes,
            'packets_per_second': packets_per_second,
            'bytes_per_second': bytes_per_second,
            'avg_packet_size': avg_packet_size,
            'syn_count': syn_count,
            'ack_count': ack_count,
            'fin_count': fin_count,
            'rst_count': rst_count,
            'start_time': start_time,
            'end_time': end_time
        }

        flow_features.append(flow_feat)

    df = pd.DataFrame(flow_features)
    print(f"[+] Creat DataFrame cu {len(df)} flows")

    return df


# ============================================================================
# SECȚIUNEA 4: Export și salvare date
# ============================================================================

def save_to_csv(df, output_path):
    """
    Salvează DataFrame-ul într-un fișier CSV.

    Args:
        df: DataFrame pandas
        output_path: Calea pentru fișierul output
    """
    df.to_csv(output_path, index=False)
    print(f"\n[+] Date salvate în: {output_path}")
    print(f"    Dimensiune: {len(df)} rânduri x {len(df.columns)} coloane")


def print_statistics(df):
    """
    Afișează statistici despre DataFrame.

    Args:
        df: DataFrame pandas
    """
    print("\n" + "=" * 60)
    print("STATISTICI DATASET")
    print("=" * 60)

    print(f"\nDimensiune: {df.shape[0]} rânduri x {df.shape[1]} coloane")

    print("\nColoane disponibile:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    if 'protocol' in df.columns:
        print("\nDistribuție protocoale:")
        print(df['protocol'].value_counts())

    if 'packet_count' in df.columns:
        print("\nStatistici flow-uri:")
        print(df[['packet_count', 'total_bytes', 'duration']].describe())


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Funcția principală - demonstrează fluxul complet de procesare.
    """
    print("=" * 60)
    print("LABORATOR 2: Procesarea datelor PCAP")
    print("=" * 60)

    # Verificăm dacă avem un fișier PCAP ca argument
    if len(sys.argv) > 1:
        pcap_file = sys.argv[1]
    else:
        # Folosim un fișier implicit pentru demo, relativ la locația scriptului
        pcap_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "pcap_samples", "http_traffic.pcap")
        )

        # Dacă nu există, creăm date de test
        if not os.path.exists(pcap_file):
            print("\n[!] Fișierul PCAP nu a fost găsit.")
            print("[*] Generăm date de test...")

            from scapy.all import IP, TCP, UDP, DNS, DNSQR, wrpcap, RandShort
            import random

            # Generăm câteva pachete de test
            test_packets = []
            for i in range(50):
                # Pachete TCP
                pkt = IP(src=f"192.168.1.{random.randint(1,254)}",
                        dst="93.184.216.34") / TCP(
                    sport=random.randint(1024, 65535),
                    dport=80,
                    flags='S'
                )
                test_packets.append(pkt)

            os.makedirs(os.path.dirname(pcap_file), exist_ok=True)
            wrpcap(pcap_file, test_packets)
            print(f"[+] Creat fișier de test: {pcap_file}")

    # #PASUL 1: Încărcăm pachetele
    # packets = load_pcap(pcap_file)

    # # # PASUL 2: Explorăm câteva pachete
    # explore_packets(packets, num_packets=3)

    # #PASUL 3: Convertim pachetele în DataFrame (nivel pachet)
    # df_packets = packets_to_dataframe(packets)

    # print("\nPrimele 5 rânduri (nivel pachet):")
    # print(df_packets.head())

    # print("\nDistribuție protocoale (nivel PACHET):")
    # print(df_packets['protocol'].value_counts())

    # #print(df_packets['ip_len'].agg(['min', 'mean', 'max']))

    # # #PASUL 4: Agregăm în flows
    # df_flows = aggregate_flows(packets)

    # zero_dur = df_flows[df_flows['duration'] == 0]
    # print(f"\nFlows cu duration=0: {len(zero_dur)} din {len(df_flows)}")

    # print("\nPrimele 5 rânduri (nivel flow):")
    # print(df_flows.head())
    
    # print("\nTop 10 flows după packet_count:")
    # print(df_flows.sort_values('packet_count', ascending=False).head(10)[['ip_src','src_port','ip_dst','dst_port','protocol','packet_count','total_bytes','duration']])

    # #Detectarea anomaliilor simple
    # port_scan_threshold = 10
    # connections_per_ip = df_packets.groupby('ip_src')['dst_port'].nunique()
    # potential_scanners = connections_per_ip[connections_per_ip > port_scan_threshold]
    # print("\nPotențiali scaneri (IP-uri cu > 10 porturi diferite accesate):")
    # print(potential_scanners if len(potential_scanners) > 0 else "  Niciun scanner detectat")

    # flood_threshold = 100
    # high_rate_flows = df_flows[df_flows['packets_per_second'] > flood_threshold]
    # print(f"\nFluxuri cu rată mare (> {flood_threshold} pachete/sec): {len(high_rate_flows)}")
    # if len(high_rate_flows) > 0:
    #     print(high_rate_flows[['ip_src', 'ip_dst', 'protocol', 'packet_count', 'packets_per_second']])


    # #PASUL 5: Afișăm statistici
    # print_statistics(df_flows)


    # #PASUL 6: Salvăm în CSV
    # output_packets = pcap_file.replace('.pcap', '_packets.csv')
    # output_flows = pcap_file.replace('.pcap', '_flows.csv')

    # save_to_csv(df_packets, output_packets)
    # save_to_csv(df_flows, output_flows)

    # print("\n" + "=" * 60)
    # print("PROCESARE COMPLETĂ!")
    # print("=" * 60)
    # print(f"\nFișiere generate:")
    # print(f"  - {output_packets} (features per pachet)")
    # print(f"  - {output_flows} (features per flow)")

    # return df_packets, df_flows
    


if __name__ == "__main__":
    main()


# În VS Code:

# Ctrl + K, Ctrl + C → comentează

# Ctrl + K, Ctrl + U → decomentează

# Ctrl + / → Amândouă, dacă merge
