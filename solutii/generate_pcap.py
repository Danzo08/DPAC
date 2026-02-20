#!/usr/bin/env python3
r"""
Generator de fișiere PCAP pentru Laboratorul de Cybersecurity

Scenarii generate:
- Trafic normal (HTTP GET + DNS)
- HTTP traffic cu login (POST /login cu username+password în clar)
- Port scanning (SYN scan)
- SYN flood (DoS la nivel TCP)

Utilizare:
    python generate_pcap.py --type normal --output normal_traffic.pcap
    python generate_pcap.py --type http_traffic --output http_traffic.pcap
    python generate_pcap.py --type portscan --output port_scan.pcap
    python generate_pcap.py --type synflood --output syn_flood.pcap

    Ruleaza:
    
    //python path\generate_pcap.py --all --output-dir "path\data\pcap_samples"

Dependențe:
    pip install scapy
"""

import argparse
import random

try:
    from scapy.all import (
        IP, TCP, UDP, DNS, DNSQR, Raw,
        wrpcap, RandShort
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("AVERTISMENT: Scapy nu este instalat.py")


def generate_normal_traffic(num_packets=100):
    packets = []

    client_ip = "192.168.1.100"
    server_ip = "93.184.216.34"
    dns_server = "8.8.8.8"

    for i in range(num_packets // 5):
        dns_query = IP(src=client_ip, dst=dns_server) / UDP(sport=RandShort(), dport=53) / DNS(rd=1, qd=DNSQR(qname=f"example{i}.com"))
        dns_response = IP(src=dns_server, dst=client_ip) / UDP(sport=53, dport=RandShort()) / DNS(qr=1, aa=1, qd=DNSQR(qname=f"example{i}.com"))
        packets += [dns_query, dns_response]

    for i in range(num_packets // 3):
        src_port = random.randint(49152, 65535)
        seq = random.randint(1000000, 9999999)
        srv_seq = random.randint(1000000, 9999999)

        syn = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags='S', seq=seq)
        syn_ack = IP(src=server_ip, dst=client_ip) / TCP(sport=80, dport=src_port, flags='SA', seq=srv_seq, ack=seq + 1)
        ack = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags='A', seq=seq + 1, ack=srv_seq + 1)

        http = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags='PA', seq=seq + 1, ack=srv_seq + 1) / Raw(load=(
            f"GET /page{i}.html HTTP/1.1\r\n"
            f"Host: example.com\r\n"
            f"User-Agent: Mozilla/5.0\r\n"
            f"\r\n"
        ))

        packets += [syn, syn_ack, ack, http]

    return packets


def generate_http_traffic(num_sessions=5):
    packets = []

    client_ip = "192.168.1.100"
    server_ip = "93.184.216.34"

    for i in range(num_sessions):
        src_port = random.randint(49152, 65535)
        c_seq = random.randint(1000000, 9999999)
        s_seq = random.randint(1000000, 9999999)

        syn = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags="S", seq=c_seq)
        syn_ack = IP(src=server_ip, dst=client_ip) / TCP(sport=80, dport=src_port, flags="SA", seq=s_seq, ack=c_seq + 1)
        ack = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags="A", seq=c_seq + 1, ack=s_seq + 1)
        packets += [syn, syn_ack, ack]

        get_req = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags="PA", seq=c_seq + 1, ack=s_seq + 1) / Raw(load=(
            "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        ))
        packets.append(get_req)

        body = f"username=admin{i}&password=1234{i}"
        post_req = IP(src=client_ip, dst=server_ip) / TCP(sport=src_port, dport=80, flags="PA", seq=c_seq + 1 + len(get_req[Raw].load), ack=s_seq + 1) / Raw(load=(
            "POST /login HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
            f"{body}"
        ))
        packets.append(post_req)

    return packets


def generate_port_scan(num_ports=50):
    packets = []

    attacker_ip = "10.0.0.50"
    target_ip = "192.168.1.10"

    for port in random.sample(range(1, 65535), num_ports):
        syn = IP(src=attacker_ip, dst=target_ip) / TCP(sport=RandShort(), dport=port, flags='S', seq=random.randint(1000000, 9999999))
        packets.append(syn)

    return packets


def generate_syn_flood(num_packets=500):
    packets = []
    target_ip = "192.168.1.100"

    for _ in range(num_packets):
        src_ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        syn = IP(src=src_ip, dst=target_ip) / TCP(sport=RandShort(), dport=80, flags='S', seq=random.randint(1000000, 9999999))
        packets.append(syn)

    return packets


def save_pcap(packets, filename):
    wrpcap(filename, packets)
    print(f"[+] Salvat {len(packets)} pachete în {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generator PCAP pentru laborator cybersecurity")
    parser.add_argument('--type', '-t', choices=['normal', 'http_traffic', 'portscan', 'synflood'])
    parser.add_argument('--output', '-o', default='output.pcap')
    parser.add_argument('--count', '-c', type=int, default=100)
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--output-dir', '-d', default='.')

    args = parser.parse_args()

    if not SCAPY_AVAILABLE:
        print("Instalează scapy: pip install scapy")
        return 1

    if args.all:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        save_pcap(generate_normal_traffic(args.count), f"{args.output_dir}/normal_traffic.pcap")
        save_pcap(generate_http_traffic(max(1, args.count // 20)), f"{args.output_dir}/http_traffic.pcap")
        save_pcap(generate_port_scan(50), f"{args.output_dir}/port_scan.pcap")
        save_pcap(generate_syn_flood(args.count * 5), f"{args.output_dir}/syn_flood.pcap")

    elif args.type:
        if args.type == 'normal':
            packets = generate_normal_traffic(args.count)
        elif args.type == 'http_traffic':
            packets = generate_http_traffic(max(1, args.count))
        elif args.type == 'portscan':
            packets = generate_port_scan(args.count)
        elif args.type == 'synflood':
            packets = generate_syn_flood(args.count)

        save_pcap(packets, args.output)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
