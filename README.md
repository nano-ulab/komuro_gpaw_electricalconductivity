# Scripts for electric conductivity calculation by GPAW

## 概要

[GPAWライブラリ](https://wiki.fysik.dtu.dk/gpaw/)([ソースコード](https://gitlab.com/gpaw/gpaw))を使用し，第一原理計算と非平衡グリーン関数法(NEGF法)を用いた対象構造の電気伝導度計算を行います．
計算時間は標準的な設定(計算手法:lcao/交換相関汎関数:PBE/基底関数:dzp)の場合，約500電子で30分程(4-core)かかります．

現在，電気伝導度計算を行う際に用いる，get_lcao_hamiltonian()関数を並列動作させると動かないことを確認しているため，伝導度計算は前半のSCF計算(並列化可能)と後半の透過関数・電気伝導度計算(並列化不可能)に分割して行うことを想定しています．
SCF計算で求められた系の状態は.gpwファイルに書き出され，透過関数計算用のスクリプトがそれをファイル名から特定し，計算を再開します．
簡単のため，両スクリプトを続けて動作させるシェルスクリプトを用意しているので，bashコマンド等からそのファイルを叩けば計算は自動的に実行されます．

## 各スクリプトの説明

### Python

### EC_SCF_loadcif.py

EC_SCF_loadcif.pyは，cif_filesフォルダ下の.cifファイルをloadしてSCF計算を行い，同フォルダ下に.gpwファイルをdumpするスクリプトです．引数にフォルダ名を選択でき，シェルスクリプト経由での角度毎のSCF計算の自動化が可能です．

### EC_Transmission_loadcif.py

EC_Transmission_loadcif.pyは，cif_filesフォルダ下の.gpwファイルををpickして，フェルミエネルギーと各領域のハミルトニアンを回収した後，指定の電圧範囲で透過関数(Transmission Function)及びそれに基づく伝導率計算，またそれらに対応するPDOSの計算を行った後，それらをプロットします．電流電圧特性については.csvファイルを別途dumpします．引数には対象構造名，V_low，V_high，V_deltaをとります．

現状，PDOS計算部分はコメントアウトしています．

### EC_SCF_FeC-graphene.py

EC_SCF_FeC-graphene.pyは，スクリプト内で定義したFeC-grapheneに対してSCF計算を行い，cif_files_originalフォルダ下に.gpwファイルをdumpするスクリプトです．引数に対象構造名と中央のFeCの回転角度をオイラー角で取り，FeCのgrapheneに対する相対位置をOverlap/Bridge/Shiftの3種から選択できるため，シェルスクリプト経由での角度毎のSCF計算の自動化が可能です．
但し，現時点ではFeC-grapheneの構造をスクリプト内に埋め込んでしまっているため，FeC，grapheneの官能基化には別のスクリプトを必要としているため，FeC-graphene構造定義部分は分離することが好ましいです．
また，現在はcif_files_originalフォルダ下にスクリプト内で定義した構造を.cifファイルとして出力するようにしています．

### EC_Transmission_FeC-graphene.py

EC_Transmission_FeC-graphene.pyは，cif_files_originalフォルダ下の.gpwファイルををpickして，フェルミエネルギーと各領域のハミルトニアンを回収した後，指定の電圧範囲で透過関数(Transmission Function)及びそれに基づく伝導率計算，またそれらに対応するPDOSの計算を行った後，それらをプロットします．電流電圧特性については.csvファイルを別途dumpします．引数には対象構造名，V_low，V_high，V_deltaをとります．

現状，PDOS計算部分はコメントアウトしています．

### Shell scripts

### EC_loadcif.sh

EC_loadcif.shは，EC_SCF_loadcif.pyとEC_Transmission_loadcif.pyに適当な引数を渡し，連続して動作させることでSCF計算から電気伝導度計算までを一気に行う為のシェルスクリプトです．
cifファイルを用いた計算を行いたいだけであれば，設定したフォルダ(現在はcif_files)下に対象の構造の.cifファイルを置き，このスクリプトを叩くだけで計算が実行されます．
稼働時に並列で動かすコア数を指定できるので，環境に応じて適当なコア数を設定してください．

### EC_FeC-graphene.sh

EC_FeC-graphene.shは，EC_SCF_FeC-graphene.pyとEC_Transmission_FeC-graphene.pyに適当な引数を渡し，連続して動作させることでSCF計算から電気伝導度計算までを一気に行う為のシェルスクリプトです．
稼働時に並列で動かすコア数を指定できるので，環境に応じて適当なコア数を設定してください．
