import Image from "next/image";

const author = {
    name: "Wilson Alfando",
    nim: "535220219",
    prodi: "Teknik Informatika",
    kampus: "Universitas Tarumanagara",
    image: "/author2.jpg",
};
const initials = author.name.split(" ").slice(0, 2).map((w) => w[0]).join("").toUpperCase();

const dosenUtama = {
    Role: "Dosen Pembimbing Utama",
    name: "Tri Sutrisno",
    gelar: "(S.Si., M.Sc.)",
    image: "/Tri Sutrisno, S.Si.,M.Sc NIK 10816004.jpeg",
};

const dosenPembimbing = {
    Role: "Dosen Pembimbing Pendamping",
    name: "Irvan Lewenusa",
    gelar: "(S.Kom., M.Kom.)",
    image: "/PakIrvan.jpg",
};

function PersonCard({ person }: { person: typeof dosenUtama }) {
    const initials = person.name.split(" ").slice(0, 2).map((w) => w[0]).join("").toUpperCase();
    return (
        <div className="bg-gray-50 border border-gray-100 rounded-2xl p-8 w-full max-w-lg flex flex-col sm:flex-row items-center gap-8">
            <div className="w-36 h-36 rounded-2xl overflow-hidden shrink-0 bg-indigo-600 flex items-center justify-center">
                {person.image ? (
                    <Image src={person.image} alt={person.name} width={144} height={144} className="object-cover w-full h-full" />
                ) : (
                    <span className="text-white text-4xl font-bold select-none">{initials}</span>
                )}
            </div>
            <div className="flex flex-col gap-1 sm:items-start items-center text-center sm:text-left">
                <p className="text-xl font-semibold text-gray-800">{person.name}</p>
                <p className="text-sm text-gray-500">{person.Role}</p>
                <p className="text-sm text-gray-500">{person.gelar}</p>
            </div>
        </div>
    );
}

export default function AboutPage() {
    return (
        <>
            {/* Tentang Pembuat */}
            <section className="bg-white py-14">
                <div className="max-w-6xl mx-auto px-6 flex flex-col items-center">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">Tentang Pembuat</h2>
                    <p className="text-gray-500 text-sm mb-8 text-center">Perancangan ini merupakan bagian dari skripsi</p>
                    <div className="bg-gray-50 border border-gray-100 rounded-2xl p-8 w-full max-w-lg flex flex-col sm:flex-row items-center gap-8">
                        <div className="w-36 h-36 rounded-2xl overflow-hidden shrink-0 bg-indigo-600 flex items-center justify-center">
                            {author.image ? (
                                <Image
                                    src={author.image}
                                    alt={author.name}
                                    width={144}
                                    height={144}
                                    className="object-cover w-full h-full"
                                />
                            ) : (
                                <span className="text-white text-4xl font-bold select-none">{initials}</span>
                            )}
                        </div>
                        <div className="flex flex-col gap-1 sm:items-start items-center text-center sm:text-left">
                            <p className="text-xl font-semibold text-gray-800">{author.name}</p>
                            <p className="text-sm text-gray-500">NIM: {author.nim}</p>
                            <p className="text-sm text-gray-500">{author.prodi}</p>
                            <p className="text-sm text-gray-500">{author.kampus}</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Dosen */}
            <section className="bg-gray-50 py-14">
                <div className="max-w-6xl mx-auto px-6 flex flex-col items-center">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">Dosen Pembimbing</h2>
                    <p className="text-gray-500 text-sm mb-8 text-center">Dosen yang terlibat dalam penelitian ini</p>
                    <div className="flex flex-col gap-4 w-full max-w-lg">
                        <PersonCard person={dosenUtama} />
                        <PersonCard person={dosenPembimbing} />
                    </div>
                </div>
            </section>
        </>
    );
}
