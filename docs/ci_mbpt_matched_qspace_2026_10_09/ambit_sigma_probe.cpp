// Independent upstream diagram execution, linked to the pinned AMBiT library.
// This adapter supplies identical integral values; it contains no MBPT formula.
#include "MBPT/CoreMBPTCalculator.h"
#include <array>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
namespace Ambit { Debug DebugOptions; int NumProcessors=1; int ProcessorRank=0; }
using namespace Ambit;
using Key = std::array<int,5>;
Key canonical(int k, int a, int b, int c, int d) {
    std::array<int,2> x={std::min(a,c),std::max(a,c)}, y={std::min(b,d),std::max(b,d)};
    if(y<x) std::swap(x,y);
    return {k,x[0],x[1],y[0],y[1]};
}
class InputIntegrals : public SlaterIntegralsInterface {
    std::map<Key,double> values;
    std::map<OrbitalInfo,int> ids;
    std::set<int> core_ids;
public:
    int core_count=0;
    InputIntegrals(pOrbitalManager m, const std::map<OrbitalInfo,int>& index,
                   const std::set<int>& core, const std::string& filename):
        SlaterIntegralsInterface(m,true), ids(index), core_ids(core) {
        std::ifstream in(filename); int k,a,b,c,d; double v;
        // File keys are already (k, sorted pair 13, sorted pair 24).
        while(in>>k>>a>>b>>c>>d>>v) values[{k,a,b,c,d}]=v;
        if(!in.eof()) throw std::runtime_error("invalid integral table");
    }
    unsigned int CalculateTwoElectronIntegrals(pOrbitalMapConst,pOrbitalMapConst,pOrbitalMapConst,pOrbitalMapConst,bool=false) override { return values.size(); }
    unsigned int size() const override {return values.size();}
    void clear() override {}
    void Read(const std::string&) override {throw std::runtime_error("no implicit table loading");}
    void Write(const std::string&) const override {throw std::runtime_error("no implicit table saving");}
    double GetTwoElectronIntegral(unsigned int k,const OrbitalInfo& a,const OrbitalInfo& b,const OrbitalInfo& c,const OrbitalInfo& d) const override {
        int ia=ids.at(a), ib=ids.at(b), ic=ids.at(c), id=ids.at(d);
        int count=core_ids.count(ia)+core_ids.count(ib)+core_ids.count(ic)+core_ids.count(id);
        if(core_count && count!=core_count) return 0.;
        auto kk=canonical(k,ia,ib,ic,id);
        auto found=values.find(kk);
        if(found==values.end()) {
            std::ostringstream message; message<<"missing integral";
            for(auto v:kk) message<<" "<<v;
            throw std::runtime_error(message.str());
        }
        return found->second;
    }
};
class Probe : public CoreMBPTCalculator {
public:
    using CoreMBPTCalculator::CoreMBPTCalculator;
    void argument(int k, double energy) {ValenceEnergies[k]=energy;}
};
int main(int argc, char** argv) {
    if(argc!=4) return 2;
    outstream=&std::cerr; logstream=&std::cerr; errstream=&std::cerr;
    auto lattice=std::make_shared<Lattice>(10,1e-6,10.);
    auto m=std::make_shared<OrbitalManager>(lattice);
    m->core=std::make_shared<OrbitalMap>(lattice);
    m->excited=std::make_shared<OrbitalMap>(lattice);
    m->particle=std::make_shared<OrbitalMap>(lattice);
    m->valence=std::make_shared<OrbitalMap>(lattice);
    m->all=std::make_shared<OrbitalMap>(lattice);
    std::ifstream states(argv[1]); int id,n,k,occupied; double e;
    std::map<int,OrbitalInfo> by_id;
    std::map<OrbitalInfo,int> ids;
    std::set<int> core_ids;
    while(states>>id>>n>>k>>e>>occupied) {
        OrbitalInfo info(n,k); auto o=std::make_shared<Orbital>(k,n,e);
        by_id.emplace(id,info); ids.emplace(info,id); m->all->AddState(o);
        if(occupied) {m->core->AddState(o); core_ids.insert(id);}
        else {m->excited->AddState(o); m->particle->AddState(o); m->valence->AddState(o);}
    }
    m->MakeStateIndexes();
    auto table=std::make_shared<InputIntegrals>(m,ids,core_ids,argv[2]);
    auto one=std::make_shared<HFIntegrals>(m,nullptr);
    Probe calc(m,one,table); calc.SetEnergyFloor(1e-8);
    std::ifstream requests(argv[3]); int a,b; double energy;
    std::cout<<std::setprecision(17);
    while(requests>>a>>b>>energy) {
        auto aa=by_id.at(a), bb=by_id.at(b); calc.argument(aa.Kappa(),energy);
        table->core_count=1; double pp=calc.GetOneElectronDiagrams(aa,bb);
        table->core_count=2; double hh=calc.GetOneElectronDiagrams(aa,bb);
        table->core_count=0; double total=calc.GetOneElectronDiagrams(aa,bb);
        std::cout<<a<<" "<<b<<" "<<energy<<" "<<pp<<" "<<hh<<" "<<total<<"\n";
    }
}
