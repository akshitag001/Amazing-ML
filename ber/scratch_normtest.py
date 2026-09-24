import sys; sys.stdout.reconfigure(encoding="utf-8")
from normalize import norm_name, norm_addr
names=[("Veoaria Sys d/b/a Heartland Online LLC","US"),("heartlandonline.com","US"),("Heartland Online L.L.C. (ID: 10156)","US"),
("Heartland Ónline (LLC)","US"),("RAMRAJ lNDIA PRÍVATE LIMITED","India"),("re1iablesilversquare.com","US"),("Tejaksh 0utsourcing Private (Limited)","India"),
("Safe Plumbing - 6404788118","US"),("Bombay Developers Piatve Limited #50543","India"),("Gpa Net Pfrivate (Limited)","India"),
("Ratna Safety (India) Pvt Ltd. | www.ratnasafe.com","India"),("Zetayuma a/k/a Amicale du Folle","France"),("Jardins Lycee [S.A.S.]","France"),
("DUNKERQUE SPORTIVE E.U.R.L.","France"),("Joueurs  Union (France) EURL","France"),("*** Birch Corp","US"),("[(L.L.P.)] ABHlNAVA PET (INDIA)","India"),
("Dr Prakasam Consultech Private Private Limited","India"),("स्मार्ट फूड्स प्राइवेट लिमिटेड","India"),("Smart Foods Private Limited","India"),
("सिल्वर कंस्ट्रक्शन लिमिटेड","India"),("Silver Construction Limited","India"),("ബാലാജി ടെക്നോളജി പ്രൈവറ്റ് ലിമിറ്റഡ്","India"),("Balaji Technology Private Limited","India"),
("লাইফ ইন্টারন্যাশনাল প্রাইভেট লিমিটেড","India"),("Torres and Inc. Brown","US"),("B+ Retail Inc","US"),
# synthetic stress: RTL, CJK, Cyrillic, mixed, emoji, control chars, empty
("شركة النور للتجارة ذ.م.م","Egypt"),("北京华为技术有限公司","China"),("株式会社トヨタ","Japan"),("ООО «Ромашка»","Russia"),
("Café ☕ Ünïcödé 😀 GmbH","Germany"),("\u200fمطعم\u200e 123\x00\t","X"),("",""),("   ",""),("ĄĆĘŁŃÓŚŹŻ Sp. z o.o.","Poland"),("삼성전자 주식회사","Korea")]
for n,c in names:
    r=norm_name(n,c); print(f"{n!r:55} -> full={r['name_full']!r} core={r['name_core']!r} legal={r['name_legal']!r} alias={r['name_alias']!r} skel={r['name_skel']!r}")
print()
addrs=["22 Green Field Pl, Texas, Sping","22 Green Field Place, Spring, TX","KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi","407/7, CHETAK CENTRE, R.N.T. MARG, Madhya Pradesh",
"Door No 881 407, R.n.t. Marg, MP","3377-C 207TH AVE E, BROKEN ARROW, OK","Broken Arrow, 337 207rd Avenue East, Oklahoma","145 Peabody Road, N/A, Vacaville, California",
"175 EDGEVALE RD, NULL, COLUMBUS, OH","Plot No. A-8/14-2, Khed, Pune, महाराष्ट्र","H.NO 05 , SOFTWARE UNITS LAYOUT, MADHAPUR, HYDERABAD, తెలంగాణ",
"N°29 Avenue Des Eiedrs Cap Ferret, Lège-Cap-Ferret","11 Rue Des Debris St Etienne, Lille","42 R. DE ROCHDALE, Tourcoing, Nord","Calais, 188 R. Dugray Trouin, Hauts-de-France",
"Shop 4, Near SBI ATM, MG Road, Indore, MP","None","NULL","","北京市朝阳区建国路88号","ул. Тверская, д. 7, Москва","شارع التحرير 15، القاهرة"]
for a in addrs: print(f"{a!r:60} -> {norm_addr(a)}")
