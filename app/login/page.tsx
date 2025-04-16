tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function LoginPage() {
  const [id, setId] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const router = useRouter();

  const handleIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setId(e.target.value);
  }
  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  }

  const handleSubmit = (e:React.FormEvent) => {
    e.preventDefault();
    // Add authentication logic here (e.g., check against hardcoded credentials)
    if (id === "admin" && password === "password") {
      router.push("/notice/write");
    } else {
        alert("Invalid id or password");
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center">
      <main className="flex-1">
        <section className="py-16">
          <div className="container">
            <div className="text-center">
              <h1 className="text-2xl font-bold text-navy-700">로그인</h1>
            </div>

            <form className="mt-8 max-w-md space-y-6" onSubmit={handleSubmit}>
              <div>
                <label className="block font-medium text-gray-700">아이디</label>
                <Input
                  type="text"
                  value={id}
                  onChange={handleIdChange}
                  className="block w-full mt-2 border-gray-300 rounded-md shadow-sm focus:border-navy-300 focus:ring focus:ring-navy-200 focus:ring-opacity-50"
                />
              </div>
              <div>
                <label className="block font-medium text-gray-700">비밀번호</label>
                <Input
                  type="password"
                  value={password}
                  onChange={handlePasswordChange}
                  className="block w-full mt-2 border-gray-300 rounded-md shadow-sm focus:border-navy-300 focus:ring focus:ring-navy-200 focus:ring-opacity-50"
                />
              </div>
              <div className="mt-8 flex justify-center">
                  <Button type="submit" className="bg-navy-600 hover:bg-navy-700 text-white py-2 px-4 rounded-md">로그인</Button>
              </div>
            </form>
          </div>
        </section>
      </main>
    </div>
  );
}